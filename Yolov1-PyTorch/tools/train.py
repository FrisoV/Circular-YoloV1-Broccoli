import torch
import argparse
import json
import math as _math
import os
import shutil
import tempfile
import numpy as np
import yaml
import random
from models.yolo import YOLOV1
from tqdm import tqdm
from dataset.voc import VOCDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, SequentialLR, LinearLR
from loss.yolov1_loss import YOLOV1Loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def state_dict_is_finite(state_dict):
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
            return False
    return True


def gradients_are_finite(model):
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return False
    return True


def estimate_state_dict_size_bytes(state_dict):
    total_bytes = 0
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor):
            total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes


def save_state_dict_atomic(state_dict, checkpoint_path):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    estimated_size = estimate_state_dict_size_bytes(state_dict)
    free_space = shutil.disk_usage(checkpoint_dir).free
    safety_margin = max(estimated_size // 2, 100 * 1024 * 1024)
    if free_space < estimated_size + safety_margin:
        raise RuntimeError(
            f'Not enough free disk space to save checkpoint to {checkpoint_path}. '
            f'Estimated size: {estimated_size / (1024 ** 2):.1f} MB, '
            f'free space: {free_space / (1024 ** 2):.1f} MB.'
        )

    fd, tmp_path = tempfile.mkstemp(prefix='.checkpoint-', suffix='.tmp', dir=checkpoint_dir)
    os.close(fd)
    try:
        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, checkpoint_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


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


def _circle_iou_scalar(det, gt):
    cx1, cy1, r1 = det[0], det[1], max(det[2], 0.0)
    cx2, cy2, r2 = gt[0], gt[1], max(gt[2], 0.0)
    d = _math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    area1, area2 = _math.pi * r1 * r1, _math.pi * r2 * r2
    eps = 1e-7
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        inter = _math.pi * min(r1, r2) ** 2
    else:
        cos1 = max(-1.0, min(1.0, (d*d + r1*r1 - r2*r2) / (2*d*r1 + eps)))
        cos2 = max(-1.0, min(1.0, (d*d + r2*r2 - r1*r1) / (2*d*r2 + eps)))
        part3 = 0.5 * _math.sqrt(max(0.0, (-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2)))
        inter = r1*r1 * _math.acos(cos1) + r2*r2 * _math.acos(cos2) - part3
    return inter / max(area1 + area2 - inter, eps)


def _run_circle_val_map(model, val_loader, model_config, train_config, idx2label):
    """Compute mAP@0.5 on the val set using circle IoU. Returns a float in [0, 1]."""
    S = model_config['S']
    B = model_config['B']
    use_sigmoid = model_config['use_sigmoid']
    conf_thr = train_config.get('eval_conf_threshold', 0.001)
    nms_thr = train_config.get('nms_threshold', 0.6)
    C = len(idx2label)

    all_preds, all_gts = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            im = batch['image'].float().to(device)
            gt_circles_list = batch.get('circles', [])
            gt_labels_list = batch['labels']
            out = model(im)

            for b in range(im.shape[0]):
                if out.dim() == 4:
                    pred = out[b].clone()
                    actual_S = pred.shape[0]
                else:
                    pred = out[b].reshape(S, S, 4 * B + C).clone()
                    actual_S = S

                if use_sigmoid:
                    pred[..., :4 * B] = torch.sigmoid(pred[..., :4 * B])
                pred = torch.clamp(pred, 0., 1.)

                class_score, class_idx = torch.max(pred[..., 4 * B:], dim=-1)
                sx = torch.arange(actual_S, dtype=torch.float32, device=pred.device) / float(actual_S)
                sy = torch.arange(actual_S, dtype=torch.float32, device=pred.device) / float(actual_S)
                sy, sx = torch.meshgrid(sy, sx, indexing='ij')

                circles_list, scores_list, labels_list = [], [], []
                for k in range(B):
                    cx = (pred[..., k*4] / actual_S + sx).reshape(-1)
                    cy = (pred[..., k*4+1] / actual_S + sy).reshape(-1)
                    r  = torch.square(pred[..., k*4+2]).reshape(-1)
                    sc = (pred[..., k*4+3] * class_score).reshape(-1)
                    circles_list.append(torch.stack([cx, cy, r], dim=-1))
                    scores_list.append(sc)
                    labels_list.append(class_idx.reshape(-1))

                circles_t = torch.cat(circles_list)
                scores_t  = torch.cat(scores_list)
                labels_t  = torch.cat(labels_list)

                keep = scores_t > conf_thr
                circles_t, scores_t, labels_t = circles_t[keep], scores_t[keep], labels_t[keep]

                if circles_t.shape[0] > 0:
                    sq = torch.stack([
                        circles_t[:, 0] - circles_t[:, 2],
                        circles_t[:, 1] - circles_t[:, 2],
                        circles_t[:, 0] + circles_t[:, 2],
                        circles_t[:, 1] + circles_t[:, 2],
                    ], dim=1)
                    mask = torch.zeros(circles_t.shape[0], dtype=torch.bool, device=pred.device)
                    for cls_id in torch.unique(labels_t):
                        ci = torch.where(labels_t == cls_id)[0]
                        kept = torch.ops.torchvision.nms(sq[ci], scores_t[ci], nms_thr)
                        mask[ci[kept]] = True
                    circles_t = circles_t[mask]
                    scores_t = scores_t[mask]
                    labels_t = labels_t[mask]

                pred_dict = {v: [] for v in idx2label.values()}
                for i in range(circles_t.shape[0]):
                    lbl = idx2label[int(labels_t[i].item())]
                    pred_dict[lbl].append(circles_t[i].tolist() + [scores_t[i].item()])
                all_preds.append(pred_dict)

                gt_dict = {v: [] for v in idx2label.values()}
                gt_c = gt_circles_list[b]
                gt_l = gt_labels_list[b]
                for i in range(gt_c.shape[0]):
                    gt_dict[idx2label[int(gt_l[i].item())]].append(gt_c[i].tolist())
                all_gts.append(gt_dict)

    gt_labels_set = sorted({cls for gd in all_gts for cls in gd})
    aps = []
    for label in gt_labels_set:
        cls_dets = [(im_i, det) for im_i, pd in enumerate(all_preds)
                    if label in pd for det in pd[label]]
        cls_dets.sort(key=lambda k: -k[1][-1])
        num_gts = sum(len(gd.get(label, [])) for gd in all_gts)
        if num_gts == 0:
            continue
        gt_matched = [[False] * len(gd.get(label, [])) for gd in all_gts]
        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)
        for di, (im_i, det) in enumerate(cls_dets):
            best_iou, best_j = -1, -1
            for j, gt in enumerate(all_gts[im_i].get(label, [])):
                iou = _circle_iou_scalar(det[:3], gt)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.5 and best_j >= 0 and not gt_matched[im_i][best_j]:
                gt_matched[im_i][best_j] = True
                tp[di] = 1
            else:
                fp[di] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        eps = np.finfo(np.float32).eps
        rec = tp / max(num_gts, eps)
        prec = tp / np.maximum(tp + fp, eps)
        rec  = np.concatenate(([0.], rec,  [1.]))
        prec = np.concatenate(([0.], prec, [0.]))
        for i in range(prec.size - 1, 0, -1):
            prec[i - 1] = max(prec[i - 1], prec[i])
        idx_ch = np.where(rec[1:] != rec[:-1])[0]
        aps.append(np.sum((rec[idx_ch + 1] - rec[idx_ch]) * prec[idx_ch + 1]))
    return float(np.mean(aps)) if aps else 0.0


def train(args):
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

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    voc = VOCDataset('train',
                     im_sets=dataset_config['train_im_sets'],
                     im_size=dataset_config['im_size'],
                     S=model_config['S'],
                     B=model_config['B'],
                     C=dataset_config['num_classes'],
                     use_circles=model_config.get('use_circles', False))
    train_dataset = DataLoader(voc,
                               batch_size=train_config['batch_size'],
                               shuffle=True,
                               collate_fn=collate_function)
    
    # Create validation dataset
    voc_val = VOCDataset('val',
                         im_sets=dataset_config['val_im_sets'],
                         im_size=dataset_config['im_size'],
                         S=model_config['S'],
                         B=model_config['B'],
                         C=dataset_config['num_classes'],
                         use_circles=model_config.get('use_circles', False))
    val_dataset = DataLoader(voc_val,
                             batch_size=train_config['batch_size'],
                             shuffle=False,
                             collate_fn=collate_function)

    yolo_model = YOLOV1(im_size=dataset_config['im_size'],
                        num_classes=dataset_config['num_classes'],
                        model_config=model_config)
    yolo_model.train()
    yolo_model.to(device)
    ckpt_path = os.path.abspath(os.path.join(train_config['task_name'],
                                              train_config['ckpt_name']))
    if os.path.exists(ckpt_path):
        print(f'Loading checkpoint from {ckpt_path}')
        checkpoint_state = torch.load(ckpt_path, map_location=device)
        if state_dict_is_finite(checkpoint_state):
            yolo_model.load_state_dict(checkpoint_state)
        else:
            print(f'Checkpoint at {ckpt_path} contains non-finite values; training from scratch instead')
    else:
        print(f'No checkpoint found at {ckpt_path}, training from scratch')
    ckpt_dir = os.path.abspath(train_config['task_name'])
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Created checkpoint directory: {ckpt_dir}")

    log_fname = train_config.get('log_fname', 'training_log.json')
    log_path = os.path.join(ckpt_dir, log_fname)
    eval_map_interval = train_config.get('eval_map_interval', 0)
    training_log = []
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                training_log = json.load(f)
            print(f'Resumed training log from {log_path} ({len(training_log)} epochs)')
        except Exception as e:
            print(f'Warning: could not load existing training log: {e}')

    optimizer = torch.optim.SGD(lr=train_config['lr'],
                                params=filter(lambda p: p.requires_grad,
                                              yolo_model.parameters()),
                                weight_decay=5E-4,
                                momentum=0.9)


    # Learning rate scheduling with warmup
    warmup_epochs = train_config.get('warmup_epochs', 3)
    use_cosine = train_config.get('use_cosine_scheduler', True)
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    
    if use_cosine:
        # Warmup: gradually increase LR from 0 to target LR
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        # Cosine annealing: smooth decay after warmup
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=train_config['lr'] * 0.01)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        print(f"Using Cosine Annealing LR with {warmup_epochs} warmup epochs")
    else:
        # Fallback to MultiStepLR
        scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.5)
        print(f"Using MultiStepLR with milestones at {train_config['lr_steps']}")
    
    criterion = YOLOV1Loss(S=model_config['S'],
                           B=model_config['B'],
                           C=dataset_config['num_classes'],
                           use_circles=model_config.get('use_circles', False))
    steps = 0
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = train_config.get('early_stop_patience', num_epochs)  # No early stop by default
    
    for epoch_idx in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\n=== Epoch {epoch_idx+1}/{num_epochs} | LR: {current_lr:.6f} ===')
        # Training phase
        yolo_model.train()
        train_losses = []
        optimizer.zero_grad()
        for idx, batch in enumerate(tqdm(train_dataset, desc=f'Epoch {epoch_idx+1}/{num_epochs} - Training')):
            im = batch['image'].float().to(device)
            yolo_targets = batch['yolo_targets'].float().to(device)
            yolo_preds = yolo_model(im)
            loss = criterion(yolo_preds, yolo_targets, use_sigmoid=model_config['use_sigmoid'])
            if not torch.isfinite(loss):
                # Save debug info for offline inspection
                debug_dir = os.path.join(ckpt_dir, 'debug_nonfinite')
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(debug_dir, f'epoch{epoch_idx+1}_batch{idx}.pt')
                try:
                    torch.save({
                        'image': im.detach().cpu(),
                        'yolo_targets': yolo_targets.detach().cpu(),
                        'yolo_preds': yolo_preds.detach().cpu(),
                        'loss': loss.detach().cpu()
                    }, debug_path)
                    print(f'Non-finite loss at batch {idx}; debug saved to {debug_path}; skipping this batch')
                except Exception as e:
                    print(f'Non-finite loss at batch {idx}; failed to save debug ({e}); skipping this batch')
                optimizer.zero_grad()
                continue
            loss = loss / acc_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(yolo_model.parameters(), max_norm=1.0) # Added gradient clipping to prevent early exitting due to NaN loss
            train_losses.append(loss.item())
            if (idx + 1) % acc_steps == 0:
                if gradients_are_finite(yolo_model):
                    optimizer.step()
                else:
                    print(f'Non-finite gradients at batch {idx}; skipping optimizer step')
                optimizer.zero_grad()
            if steps % train_config['log_steps'] == 0:
                print('Train Loss : {:.4f}'.format(np.mean(train_losses)))
            steps += 1

        # Flush remainder gradients when gradient accumulation is used.
        if len(train_dataset) % acc_steps != 0:
            if gradients_are_finite(yolo_model):
                optimizer.step()
            else:
                print('Non-finite gradients at epoch end; skipping optimizer step')
            optimizer.zero_grad()
        
        # Validation phase
        yolo_model.eval()
        val_losses = []
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_dataset, desc=f'Epoch {epoch_idx+1}/{num_epochs} - Validation')):
                im = batch['image'].float().to(device)
                yolo_targets = batch['yolo_targets'].float().to(device)
                yolo_preds = yolo_model(im)
                loss = criterion(yolo_preds, yolo_targets, use_sigmoid=model_config['use_sigmoid'])
                if not torch.isfinite(loss):
                    # Save debug info for offline inspection
                    debug_dir = os.path.join(ckpt_dir, 'debug_nonfinite')
                    os.makedirs(debug_dir, exist_ok=True)
                    debug_path = os.path.join(debug_dir, f'val_epoch{epoch_idx+1}_batch{idx}.pt')
                    try:
                        torch.save({
                            'image': im.detach().cpu(),
                            'yolo_targets': yolo_targets.detach().cpu(),
                            'yolo_preds': yolo_preds.detach().cpu(),
                            'loss': loss.detach().cpu()
                        }, debug_path)
                        print(f'Non-finite val loss at batch {idx}; debug saved to {debug_path}; skipping this batch')
                    except Exception as e:
                        print(f'Non-finite val loss at batch {idx}; failed to save debug ({e}); skipping this batch')
                    continue
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses) * acc_steps  # Multiply back since we divided during training
        avg_val_loss = np.mean(val_losses)
        
        print(f'\nEpoch {epoch_idx+1}/{num_epochs} Summary:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss:   {avg_val_loss:.4f}')
        print(f'  LR:         {current_lr:.6f}')

        # Per-epoch mAP evaluation (circle IoU)
        val_map = None
        if eval_map_interval > 0 and (epoch_idx + 1) % eval_map_interval == 0:
            print('  Running val mAP@0.5 (circle IoU)...')
            val_map = _run_circle_val_map(
                yolo_model, val_dataset, model_config, train_config, voc_val.idx2label)
            yolo_model.train()
            print(f'  Val mAP@0.5: {val_map:.4f}')

        # Save latest checkpoint
        latest_state = yolo_model.state_dict()
        if state_dict_is_finite(latest_state):
            save_state_dict_atomic(latest_state, os.path.join(ckpt_dir, train_config['ckpt_name']))
        else:
            print('Model contains non-finite values; skipping latest checkpoint save')
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss - train_config.get('early_stop_min_delta', 0.0):
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_ckpt_path = os.path.join(ckpt_dir, train_config.get('best_ckpt_name', 'best_model.pth'))
            if state_dict_is_finite(latest_state):
                save_state_dict_atomic(latest_state, best_ckpt_path)
                print(f'  ✓ Best model saved! (Val Loss: {best_val_loss:.4f})')
            else:
                print('Model contains non-finite values; skipping best checkpoint save')
        else:
            patience_counter += 1
            print(f'  No improvement for {patience_counter} epoch(s) (Best: {best_val_loss:.4f})')
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f'\nEarly stopping triggered after {epoch_idx+1} epochs (patience: {early_stop_patience})')
            break

        # Append epoch metrics to JSON log
        record = {
            'epoch': epoch_idx + 1,
            'train_loss': float(avg_train_loss),
            'val_loss': float(avg_val_loss),
            'lr': float(current_lr),
        }
        if val_map is not None:
            record['val_map50'] = float(val_map)
        training_log.append(record)
        try:
            with open(log_path, 'w') as f:
                json.dump(training_log, f, indent=2)
        except Exception as e:
            print(f'Warning: could not save training log: {e}')

        scheduler.step()
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for yolov1 training')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    args = parser.parse_args()
    train(args)
