import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import yaml
import random
from tqdm import tqdm
from models.yolo import YOLOV1
from dataset.voc import VOCDataset
from utils.visualization_utils import *
from torch.utils.data.dataloader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_function(batch):
    """Custom collate function to handle variable-sized bboxes and labels"""
    images = torch.stack([item['image'] for item in batch], dim=0)
    yolo_targets = torch.stack([item['yolo_targets'] for item in batch], dim=0)
    bboxes = [item['bboxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    file_paths = [item['file_path'] for item in batch]
    
    return {
        'image': images,
        'yolo_targets': yolo_targets,
        'bboxes': bboxes,
        'labels': labels,
        'file_path': file_paths
    }


def get_iou(det, gt):
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = float(det_area + gt_area - area_intersection + 1E-6)
    iou = area_intersection / area_union
    return iou


def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area', difficult=None):
    # det_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2, score], ...],
    #       'car' : [[x1, y1, x2, y2, score], ...]
    #   }
    #   {det_boxes_img_2},
    #   ...
    #   {det_boxes_img_N},
    # ]
    #
    # gt_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2], ...],
    #       'car' : [[x1, y1, x2, y2], ...]
    #   },
    #   {gt_boxes_img_2},
    #   ...
    #   {gt_boxes_img_N},
    # ]

    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)

    all_aps = {}
    # average precisions for ALL classes
    aps = []
    for idx, label in enumerate(gt_labels):
        # Get detection predictions of this class
        cls_dets = [
            [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for im_dets_label in im_dets[label]
        ]

        # cls_dets = [
        #   (0, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (0, [x1_M, y1_M, x2_M, y2_M, score_M]),
        #   (1, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (1, [x1_N, y1_N, x2_N, y2_N, score_N]),
        #   ...
        # ]

        # Sort them by confidence score
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])

        # For tracking which gt boxes of this class have already been matched
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        # Number of gt boxes for this class for recall calculation
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])
        num_difficults = sum([sum(difficults_label[label]) for difficults_label in difficult])

        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)

        # For each prediction
        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            # Get gt boxes for this image and this label
            im_gts = gt_boxes[im_idx][label]
            im_gt_difficults = difficult[im_idx][label]

            max_iou_found = -1
            max_iou_gt_idx = -1

            # Get best matching gt box
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            # TP only if iou >= threshold and this gt has not yet been matched
            if max_iou_found >= iou_threshold:
                if not im_gt_difficults[max_iou_gt_idx]:
                    if not gt_matched[im_idx][max_iou_gt_idx]:
                        # If tp then we set this gt box as matched
                        gt_matched[im_idx][max_iou_gt_idx] = True
                        tp[det_idx] = 1
                    else:
                        fp[det_idx] = 1
            else:
                fp[det_idx] = 1

        # Cumulative tp and fp
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        eps = np.finfo(np.float32).eps
        # recalls = tp / np.maximum(num_gts, eps)
        recalls = tp / np.maximum(num_gts - num_difficults, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))

            # Replace precision values with recall r with maximum precision value
            # of any recall value >= r
            # This computes the precision envelope
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            # For computing area, get points where recall changes value
            i = np.where(recalls[1:] != recalls[:-1])[0]
            # Add the rectangular areas to get ap
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                # Get precision values for recall values >= interp_pt
                prec_interp_pt = precisions[recalls >= interp_pt]

                # Get max of those precision values
                prec_interp_pt= prec_interp_pt.max() if prec_interp_pt.size>0.0 else 0.0
                ap += prec_interp_pt
            ap = ap / 11.0
        else:
            raise ValueError('Method can only be area or interp')
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan
    # compute mAP at provided iou threshold
    mean_ap = sum(aps) / len(aps)
    return mean_ap, all_aps


def load_model_and_dataset(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    voc = VOCDataset('test',
                     im_sets=dataset_config['test_im_sets'],
                     im_size=dataset_config['im_size'],
                     S=model_config['S'],
                     B=model_config['B'],
                     C=dataset_config['num_classes'])
    test_dataset = DataLoader(voc, batch_size=1, shuffle=False, collate_fn=collate_function)

    yolo_model = YOLOV1(im_size=dataset_config['im_size'],
                        num_classes=dataset_config['num_classes'],
                        model_config=model_config)
    yolo_model.eval()
    yolo_model.to(device)
    ckpt_name = train_config.get('best_ckpt_name', train_config['ckpt_name'])
    assert os.path.exists(os.path.join(train_config['task_name'], ckpt_name)), \
        "No checkpoint exists at {}".format(os.path.join(train_config['task_name'], ckpt_name))
    yolo_model.load_state_dict(torch.load(os.path.join(train_config['task_name'], ckpt_name),
                                          map_location=device))
    return yolo_model, voc, test_dataset, config


def convert_yolo_pred_x1y1x2y2(yolo_pred, S, B, C, use_sigmoid=False):
    r"""
    Method converts yolo predictions to
    x1y1x2y2 format
    """
    # Infer the actual grid size from the output tensor shape
    # yolo_pred shape: (batch_size, S, S, 5*B+C) for conv model or (batch_size, S*S*(5*B+C)) for fc model
    if yolo_pred.dim() == 4:
        # Conv model output: already in (batch, S, S, channels) format
        batch_size, actual_S, _, channels = yolo_pred.shape
        out = yolo_pred.squeeze(0)  # Remove batch dimension
        if actual_S != S:
            print(f"WARNING: Model outputs {actual_S}x{actual_S} grid but config specifies S={S}. Using actual size.")
        S = actual_S
    else:
        # FC model output: flatten, infer S from total size
        total_size = yolo_pred.numel()
        expected_per_spatial = 5 * B + C
        total_spatial = total_size // expected_per_spatial
        actual_S = int(total_spatial ** 0.5)
        if actual_S != S:
            print(f"WARNING: Model outputs {actual_S}x{actual_S} grid but config specifies S={S}. Using actual size.")
        S = actual_S
        out = yolo_pred.reshape((S, S, 5 * B + C))
    if use_sigmoid:
        out[..., :5 * B] = torch.nn.functional.sigmoid(out[..., :5 * B])
    out = torch.clamp(out, min=0., max=1.)
    class_score, class_idx = torch.max(out[..., 5 * B:], dim=-1)

    # Create a grid using these shifts
    # Will use these for converting x_center_offset/y_center_offset
    # values to x1/y1/x2/y2(normalized 0-1)
    # S cells = 1 => each cell adds 1/S pixels of shift
    shifts_x = torch.arange(0, S, dtype=torch.float32, device=out.device) / float(S)
    shifts_y = torch.arange(0, S, dtype=torch.float32, device=out.device) / float(S)
    shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

    boxes = []
    confidences = []
    labels = []
    for box_idx in range(B):
        # xc_offset yc_offset w h -> x1 y1 x2 y2
        boxes_x1 = ((out[..., box_idx * 5] * 1 / float(S) + shifts_x) -
                    0.5 * torch.square(out[..., 2 + box_idx * 5])).reshape(-1, 1)
        boxes_y1 = ((out[..., 1 + box_idx * 5] * 1 / float(S) + shifts_y) -
                    0.5 * torch.square(out[..., 3 + box_idx * 5])).reshape(-1, 1)
        boxes_x2 = ((out[..., box_idx * 5] * 1 / float(S) + shifts_x) +
                    0.5 * torch.square(out[..., 2 + box_idx * 5])).reshape(-1, 1)
        boxes_y2 = ((out[..., 1 + box_idx * 5] * 1 / float(S) + shifts_y) +
                    0.5 * torch.square(out[..., 3 + box_idx * 5])).reshape(-1, 1)
        boxes.append(torch.cat([boxes_x1, boxes_y1, boxes_x2, boxes_y2], dim=-1))
        confidences.append((out[..., 4 + box_idx * 5] * class_score).reshape(-1))
        labels.append(class_idx.reshape(-1))
    boxes = torch.cat(boxes, dim=0)
    scores = torch.cat(confidences, dim=0)
    labels = torch.cat(labels, dim=0)
    return boxes, scores, labels


def infer(args):
    if not os.path.exists('samples'):
        os.mkdir('samples')

    yolo_model, voc, test_dataset, config = load_model_and_dataset(args)
    conf_threshold = config['train_params']['infer_conf_threshold']
    nms_threshold = config['train_params']['nms_threshold']

    sample_indices = config['train_params'].get('infer_sample_indices', list(range(10)))

    for i, dataset_idx in tqdm(enumerate(sample_indices)):
        sample = voc[dataset_idx]
        im_tensor = sample['image']
        fname = sample['file_path']

        out = yolo_model(im_tensor.unsqueeze(0).to(device))
        boxes, scores, labels = convert_yolo_pred_x1y1x2y2(out,
                                                           S=yolo_model.S,
                                                           B=yolo_model.B,
                                                           C=yolo_model.C,
                                                           use_sigmoid=config['model_params']['use_sigmoid'])

        # Confidence Score Thresholding
        keep = torch.where(scores > conf_threshold)[0]
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # NMS
        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        for class_id in torch.unique(labels):
            curr_indices = torch.where(labels == class_id)[0]
            curr_keep_indices = torch.ops.torchvision.nms(boxes[curr_indices],
                                                          scores[curr_indices],
                                                          nms_threshold)
            keep_mask[curr_indices[curr_keep_indices]] = True
        keep = torch.where(keep_mask)[0]
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        #################
        # Visualization #
        #################

        if not os.path.exists('samples/preds'):
            os.mkdir('samples/preds')
        if not os.path.exists('samples/grid_cls'):
            os.mkdir('samples/grid_cls')

        im = cv2.imread(fname)
        h, w = im.shape[:2]
        # Scale prediction boxes x1y1x2y2 from 0-1 to 0-w and 0-h
        boxes[..., 0::2] = (w * boxes[..., 0::2])
        boxes[..., 1::2] = (h * boxes[..., 1::2])

        out_img = visualize(image=im,
                            bboxes=boxes.detach().cpu().numpy(),
                            category_ids=labels.detach().cpu().numpy(),
                            category_id_to_name=voc.idx2label,
                            scores=scores.detach().cpu().numpy())

        cv2.imwrite('samples/preds/{}_pred.jpeg'.format(i), out_img)

        # Below lines of code are only for drawing class prob map
        im = cv2.resize(im, (yolo_model.im_size, yolo_model.im_size))

        # Infer actual grid size from output
        if out.dim() == 4:
            # Conv model output: already in (batch, S, S, channels) format
            actual_S = out.shape[1]
            out_reshaped = out.squeeze(0)
        else:
            # FC model output: infer S from total size
            total_size = out.numel()
            expected_per_spatial = 5 * yolo_model.B + yolo_model.C
            total_spatial = total_size // expected_per_spatial
            actual_S = int(total_spatial ** 0.5)
            out_reshaped = out.reshape((actual_S, actual_S, 5 * yolo_model.B + yolo_model.C))

        # Draw a SxS grid on image
        grid_im = draw_grid(im, (actual_S, actual_S))

        cls_val, cls_idx = torch.max(out_reshaped[..., 5 * yolo_model.B:], dim=-1)

        # Draw colored squares for probability mappings on image
        rect_im = draw_cls_grid(im, cls_idx, (actual_S, actual_S))
        # Draw grid again on top of this image
        rect_im = draw_grid(rect_im, (actual_S, actual_S))

        # Overlay image with grid and cls mappings with grid on top of each other
        res = cv2.addWeighted(rect_im, 0.5, grid_im, 0.5, 1.0)
        # Write class labels on grid on this image
        res = draw_cls_text(res, cls_idx, voc.idx2label, (actual_S, actual_S))
        cv2.imwrite('samples/grid_cls/{}_grid_map.jpeg'.format(i), res)
    print('Done Detecting...')


def evaluate_map(args):
    yolo_model, voc, test_dataset, config = load_model_and_dataset(args)
    conf_threshold = config['train_params']['eval_conf_threshold']
    nms_threshold = config['train_params']['nms_threshold']

    gts = []
    preds = []
    difficults = []
    for batch in tqdm(test_dataset):
        im_tensor = batch['image'].float().to(device)
        target_bboxes = batch['bboxes'][0]
        target_labels = batch['labels'][0]
        # Handle difficult field if it exists
        difficult = batch.get('difficult', [torch.zeros(len(target_labels), dtype=torch.long)])[0]
        out = yolo_model(im_tensor)
        boxes, scores, labels = convert_yolo_pred_x1y1x2y2(out,
                                                           S=yolo_model.S,
                                                           B=yolo_model.B,
                                                           C=yolo_model.C,
                                                           use_sigmoid=config['model_params']['use_sigmoid'])
        # DEBUG: Check prediction scores
        #print(f"\n=== Sample Debug ===")
        #print(f"Total predictions before filtering: {len(boxes)}")
        #print(f"Max confidence score: {scores.max():.6f}")
        #print(f"Mean confidence score: {scores.mean():.6f}")
        #print(f"Predictions above 0.5: {(scores > 0.5).sum().item()}")
        #print(f"Predictions above 0.1: {(scores > 0.1).sum().item()}")
        #print(f"Predictions above 0.01: {(scores > 0.01).sum().item()}")

        # Confidence Score Thresholding
        keep = torch.where(scores > conf_threshold)[0]
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # NMS
        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        for class_id in torch.unique(labels):
            curr_indices = torch.where(labels == class_id)[0]
            curr_keep_indices = torch.ops.torchvision.nms(boxes[curr_indices],
                                                          scores[curr_indices],
                                                          nms_threshold)
            keep_mask[curr_indices[curr_keep_indices]] = True

        keep = torch.where(keep_mask)[0]
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        pred_boxes = {}
        gt_boxes = {}
        difficult_boxes = {}

        for label_name in voc.label2idx:
            pred_boxes[label_name] = []
            gt_boxes[label_name] = []
            difficult_boxes[label_name] = []

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = labels[idx].detach().cpu().item()
            score = scores[idx].detach().cpu().item()
            label_name = voc.idx2label[label]
            pred_boxes[label_name].append([x1, y1, x2, y2, score])
        for idx, box in enumerate(target_bboxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = target_labels[idx].detach().cpu().item()
            label_name = voc.idx2label[label]
            gt_boxes[label_name].append([x1, y1, x2, y2])
            difficult_boxes[label_name].append(difficult[idx].detach().cpu().item())

        gts.append(gt_boxes)
        preds.append(pred_boxes)
        difficults.append(difficult_boxes)
    mean_ap, all_aps = compute_map(preds, gts, method='area', difficult=difficults)
    print('Class Wise Average Precisions')
    for idx in range(len(voc.idx2label)):
        print('AP for class {} = {:.4f}'.format(voc.idx2label[idx],
                                                all_aps[voc.idx2label[idx]]))
    print('Mean Average Precision : {:.4f}'.format(mean_ap))


def _str2bool(v):
    return v.lower() not in ('false', '0', 'no', 'n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for yolov1 inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    parser.add_argument('--evaluate', dest='evaluate',
                        default=False, type=_str2bool)
    parser.add_argument('--infer_samples', dest='infer_samples',
                        default=True, type=_str2bool)
    args = parser.parse_args()

    with torch.no_grad():
        if args.infer_samples:
            infer(args)
        else:
            print('Not Inferring for samples as `infer_samples` argument is False')

        if args.evaluate:
            evaluate_map(args)
        else:
            print('Not Evaluating as `evaluate` argument is False')
