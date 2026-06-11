import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import yaml
import math
import numpy as np
from tqdm import tqdm
from models.yolo import YOLOV1
from dataset.voc import VOCDataset
from utils.visualization_utils import visualize
from torch.utils.data.dataloader import DataLoader
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_function(batch):
    """Custom collate function to handle variable-sized bboxes and labels"""
    images = torch.stack([item['image'] for item in batch], dim=0)
    yolo_targets = torch.stack([item['yolo_targets'] for item in batch], dim=0)
    bboxes = [item['bboxes'] for item in batch]
    circles = [item.get('circles', torch.zeros(0, 3)) for item in batch]
    labels = [item['labels'] for item in batch]
    file_paths = [item['file_path'] for item in batch]

    return {
        'image': images,
        'yolo_targets': yolo_targets,
        'bboxes': bboxes,
        'circles': circles,
        'labels': labels,
        'file_path': file_paths
    }


def get_iou(det, gt):
    """Calculate IoU between two axis-aligned boxes [x1, y1, x2, y2]."""
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
    return area_intersection / area_union


def get_circle_iou(det, gt):
    """Calculate IoU between two circles [cx, cy, r]."""
    cx1, cy1, r1 = det[0], det[1], max(det[2], 0.0)
    cx2, cy2, r2 = gt[0], gt[1], max(gt[2], 0.0)
    d = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    area1, area2 = math.pi * r1 * r1, math.pi * r2 * r2
    eps = 1e-7
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        inter = math.pi * min(r1, r2) ** 2
    else:
        cos1 = max(-1.0, min(1.0, (d*d + r1*r1 - r2*r2) / (2*d*r1 + eps)))
        cos2 = max(-1.0, min(1.0, (d*d + r2*r2 - r1*r1) / (2*d*r2 + eps)))
        part3 = 0.5 * math.sqrt(max(0.0, (-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2)))
        inter = r1*r1 * math.acos(cos1) + r2*r2 * math.acos(cos2) - part3
    return inter / max(area1 + area2 - inter, eps)


def tune_thresholds(args):
    """Test different confidence and NMS thresholds and report metrics"""
    device = globals()['device']  # local copy so the except block can rebind it

    # Read config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Load model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yolo_model = YOLOV1(im_size=dataset_config['im_size'],
                        num_classes=dataset_config['num_classes'],
                        model_config=model_config)
    yolo_model.eval()
    try:
        yolo_model.to(device)
    except torch.OutOfMemoryError:
        print("WARNING: GPU OOM, falling back to CPU for threshold tuning.")
        device = torch.device('cpu')
        yolo_model.to(device)
    
    ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    if os.path.exists(ckpt_path):
        yolo_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"No checkpoint found at {ckpt_path}")
        return
    
    use_circles = model_config.get('use_circles', False)

    # Load dataset
    voc = VOCDataset('test',
                     im_sets=dataset_config['test_im_sets'],
                     im_size=dataset_config['im_size'],
                     S=model_config['S'],
                     B=model_config['B'],
                     C=dataset_config['num_classes'],
                     use_circles=use_circles)
    test_dataset = DataLoader(voc, batch_size=1, shuffle=False, collate_fn=collate_function)
    
    # Thresholds to test
    conf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    nms_thresholds = [0.4, 0.5, 0.6, 0.7]
    
    print("\n" + "="*80)
    print("CONFIDENCE & NMS THRESHOLD TUNING")
    print("="*80)
    print("\nTesting different threshold combinations...")
    print(f"Confidence thresholds: {conf_thresholds}")
    print(f"NMS thresholds: {nms_thresholds}")
    
    results = []
    
    with torch.no_grad():
        for conf_thresh in conf_thresholds:
            for nms_thresh in nms_thresholds:
                total_detections = 0
                total_ground_truth = 0
                matched = 0
                
                for batch in tqdm(test_dataset, desc=f'Conf={conf_thresh}, NMS={nms_thresh}', leave=False):
                    im_tensor = batch['image'].float().to(device)
                    if use_circles:
                        gt_objects = batch.get('circles', [torch.zeros(0, 3)])[0]
                    else:
                        gt_objects = batch['bboxes'][0]

                    if len(gt_objects) == 0:
                        continue

                    # Forward pass
                    out = yolo_model(im_tensor)

                    if use_circles:
                        from tools.infer import convert_yolo_pred_circles, circle_nms
                        dets, scores, labels = convert_yolo_pred_circles(
                            out,
                            S=yolo_model.S,
                            B=yolo_model.B,
                            C=yolo_model.C,
                            use_sigmoid=model_config['use_sigmoid'],
                        )
                        keep = torch.where(scores > conf_thresh)[0]
                        dets, scores, labels = dets[keep], scores[keep], labels[keep]
                        if len(dets) > 0:
                            keep_mask = torch.zeros_like(scores, dtype=torch.bool)
                            for cls_id in torch.unique(labels):
                                ci = torch.where(labels == cls_id)[0]
                                ck = circle_nms(dets[ci], scores[ci], nms_thresh)
                                keep_mask[ci[ck]] = True
                            dets = dets[keep_mask]
                            scores = scores[keep_mask]
                    else:
                        from tools.infer import convert_yolo_pred_x1y1x2y2
                        dets, scores, labels = convert_yolo_pred_x1y1x2y2(
                            out,
                            S=yolo_model.S,
                            B=yolo_model.B,
                            C=yolo_model.C,
                            use_sigmoid=model_config['use_sigmoid'],
                            use_circles=False,
                        )
                        keep = torch.where(scores > conf_thresh)[0]
                        dets, scores, labels = dets[keep], scores[keep], labels[keep]
                        if len(dets) > 0:
                            keep_mask = torch.zeros_like(scores, dtype=torch.bool)
                            indices = torch.ops.torchvision.nms(dets, scores, nms_thresh)
                            keep_mask[indices] = True
                            dets = dets[keep_mask]

                    total_detections += len(dets)
                    total_ground_truth += len(gt_objects)

                    # Match each detection to the best unmatched GT (no GT counted twice)
                    iou_fn = get_circle_iou if use_circles else get_iou
                    gt_matched = [False] * len(gt_objects)
                    for det in dets:
                        best_iou, best_j = 0.5, -1
                        for j, gt in enumerate(gt_objects):
                            if gt_matched[j]:
                                continue
                            iou = iou_fn(det.cpu().numpy(), gt.numpy())
                            if iou > best_iou:
                                best_iou, best_j = iou, j
                        if best_j >= 0:
                            gt_matched[best_j] = True
                            matched += 1
                
                # Calculate metrics
                precision = matched / max(total_detections, 1)
                recall = matched / max(total_ground_truth, 1)
                f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)
                
                results.append({
                    'conf_thresh': conf_thresh,
                    'nms_thresh': nms_thresh,
                    'detections': total_detections,
                    'gt': total_ground_truth,
                    'matched': matched,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
    
    # Print results
    print("\n" + "="*120)
    print(f"{'Conf':<6} {'NMS':<6} {'Det':<6} {'GT':<6} {'Match':<6} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    print("="*120)
    
    for r in sorted(results, key=lambda x: x['f1'], reverse=True):
        print(f"{r['conf_thresh']:<6.1f} {r['nms_thresh']:<6.1f} {r['detections']:<6} {r['gt']:<6} "
              f"{r['matched']:<6} {r['precision']:<8.3f} {r['recall']:<8.3f} {r['f1']:<8.3f}")
    
    # Find best
    best = max(results, key=lambda x: x['f1'])
    print("\n" + "="*120)
    print(f"BEST CONFIG (by F1 score):")
    print(f"  Confidence Threshold: {best['conf_thresh']}")
    print(f"  NMS Threshold: {best['nms_thresh']}")
    print(f"  F1 Score: {best['f1']:.4f}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Recall: {best['recall']:.4f}")
    print("="*120 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune confidence and NMS thresholds')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    args = parser.parse_args()
    tune_thresholds(args)
