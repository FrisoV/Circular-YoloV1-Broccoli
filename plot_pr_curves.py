"""
plot_pr_curves.py
Generates overlaid precision-recall curves for:
  1. Baseline rectangular model  (NMS = 0.6, as originally tuned)
  2. Circular model, NMS = 0.6   (original setting)
  3. Circular model, NMS = 0.4   (re-tuned)

Each model runs a single forward pass; NMS is applied post-hoc so multiple
NMS thresholds can be compared without repeating inference.

Usage (from the Thesis/ directory):
    python plot_pr_curves.py
    python plot_pr_curves.py --output pr_curves.png
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

# ── default paths ────────────────────────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))
CIRCLE_DIR    = os.path.join(_HERE, 'Yolov1-PyTorch')
BASELINE_DIR  = os.path.join(_HERE, 'Yolov1-PyTorch Baseline')
CIRCLE_CKPT   = os.path.join(CIRCLE_DIR,   'results', 'yolo_broccoli_best.pth')
BASELINE_CKPT = os.path.join(BASELINE_DIR, 'results', 'yolo_broccoli_baseline_best.pth')
OUTPUT_PATH   = os.path.join(_HERE, 'pr_curves.png')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── IoU helpers ──────────────────────────────────────────────────────────────

def _circle_iou(det, gt):
    cx1, cy1, r1 = det[0], det[1], max(det[2], 0.0)
    cx2, cy2, r2 = gt[0],  gt[1],  max(gt[2],  0.0)
    d = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    a1, a2, eps = math.pi * r1 * r1, math.pi * r2 * r2, 1e-7
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        inter = math.pi * min(r1, r2) ** 2
    else:
        cos1  = max(-1., min(1., (d*d + r1*r1 - r2*r2) / (2*d*r1 + eps)))
        cos2  = max(-1., min(1., (d*d + r2*r2 - r1*r1) / (2*d*r2 + eps)))
        part3 = 0.5 * math.sqrt(max(0., (-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2)))
        inter = r1*r1 * math.acos(cos1) + r2*r2 * math.acos(cos2) - part3
    return inter / max(a1 + a2 - inter, eps)


def _circle_nms(circles, scores, iou_threshold):
    if circles.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=circles.device)
    order = scores.argsort(descending=True)
    suppressed = torch.zeros(circles.shape[0], dtype=torch.bool, device=circles.device)
    keep = []
    for i in range(len(order)):
        idx = order[i].item()
        if suppressed[idx]:
            continue
        keep.append(idx)
        ci = circles[idx].tolist()
        for j in range(i + 1, len(order)):
            jdx = order[j].item()
            if not suppressed[jdx] and _circle_iou(ci, circles[jdx].tolist()) >= iou_threshold:
                suppressed[jdx] = True
    return torch.tensor(keep, dtype=torch.long, device=circles.device)


def _box_iou(det, gt):
    xl = max(det[0], gt[0]); yt = max(det[1], gt[1])
    xr = min(det[2], gt[2]); yb = min(det[3], gt[3])
    if xr < xl or yb < yt:
        return 0.0
    inter = (xr - xl) * (yb - yt)
    return inter / float(
        (det[2]-det[0])*(det[3]-det[1]) + (gt[2]-gt[0])*(gt[3]-gt[1]) - inter + 1e-6)


# ── module isolation ─────────────────────────────────────────────────────────

def _purge_project_modules():
    prefixes = ('models', 'dataset', 'loss', 'utils', 'circle_ops')
    for key in [k for k in sys.modules if k.split('.')[0] in prefixes]:
        del sys.modules[key]


# ── PR-curve builder ─────────────────────────────────────────────────────────

def _build_pr(all_preds, all_gts, iou_fn, iou_threshold):
    """
    Standard mAP-style PR curve.
    all_preds: list of {label: [[*det, score], ...]} dicts per image.
    all_gts:   list of {label: [[*gt], ...]}          dicts per image.
    Returns {label: (rec_array, prec_array, ap_scalar)}.
    """
    results = {}
    for label in sorted({cls for gd in all_gts for cls in gd}):
        cls_dets = [
            (im_i, det)
            for im_i, pd in enumerate(all_preds)
            if label in pd
            for det in pd[label]
        ]
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
                iou = iou_fn(det[:-1], gt)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold and best_j >= 0 and not gt_matched[im_i][best_j]:
                gt_matched[im_i][best_j] = True
                tp[di] = 1
            else:
                fp[di] = 1

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        eps  = np.finfo(np.float32).eps
        rec  = tp / max(num_gts, eps)
        prec = tp / np.maximum(tp + fp, eps)

        rec  = np.concatenate(([0.], rec,  [1.]))
        prec = np.concatenate(([0.], prec, [0.]))
        for i in range(prec.size - 1, 0, -1):
            prec[i - 1] = max(prec[i - 1], prec[i])
        ch = np.where(rec[1:] != rec[:-1])[0]
        ap = float(np.sum((rec[ch + 1] - rec[ch]) * prec[ch + 1]))

        results[label] = (rec, prec, ap)
    return results


# ── Raw prediction collection ─────────────────────────────────────────────────

def _collect_circle_raw(model_dir, config_path, ckpt_path):
    """
    Single forward pass on the test set.
    Returns (raw_per_image, idx2label) where raw_per_image is a list of
    (circles_cpu, scores_cpu, labels_cpu, gt_circles_list) per image.
    """
    orig_dir = os.getcwd()
    os.chdir(model_dir)
    _purge_project_modules()
    sys.path.insert(0, model_dir)
    try:
        from models.yolo import YOLOV1
        from dataset.voc import VOCDataset
        from torch.utils.data import DataLoader

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        dcfg, mcfg, tcfg = cfg['dataset_params'], cfg['model_params'], cfg['train_params']

        voc = VOCDataset('test', im_sets=dcfg['test_im_sets'], im_size=dcfg['im_size'],
                         S=mcfg['S'], B=mcfg['B'], C=dcfg['num_classes'], use_circles=True)

        def _collate(batch):
            return {
                'image':   torch.stack([b['image'] for b in batch]),
                'circles': [b.get('circles', torch.zeros(0, 3)) for b in batch],
                'labels':  [b['labels'] for b in batch],
            }

        loader = DataLoader(voc, batch_size=1, shuffle=False, collate_fn=_collate)
        model  = YOLOV1(im_size=dcfg['im_size'], num_classes=dcfg['num_classes'],
                        model_config=mcfg)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval().to(device)

        S, B, C    = mcfg['S'], mcfg['B'], dcfg['num_classes']
        conf_thr   = tcfg.get('eval_conf_threshold', 0.001)
        use_sigmoid = mcfg['use_sigmoid']

        raw = []
        with torch.no_grad():
            for batch in tqdm(loader, desc='Circle model — collecting predictions'):
                im  = batch['image'].float().to(device)
                out = model(im)
                pred = out[0].clone() if out.dim() == 4 else out[0].reshape(S, S, 4*B+C).clone()
                aS   = pred.shape[0]

                if use_sigmoid:
                    pred[..., :4*B] = torch.sigmoid(pred[..., :4*B])
                pred = torch.clamp(pred, 0., 1.)

                cls_sc, cls_idx = torch.max(pred[..., 4*B:], dim=-1)
                sx = torch.arange(aS, dtype=torch.float32, device=device) / aS
                sy = torch.arange(aS, dtype=torch.float32, device=device) / aS
                sy, sx = torch.meshgrid(sy, sx, indexing='ij')

                cl, sl, ll = [], [], []
                for k in range(B):
                    cx = (pred[..., k*4]   / aS + sx).reshape(-1)
                    cy = (pred[..., k*4+1] / aS + sy).reshape(-1)
                    r  = torch.square(pred[..., k*4+2]).reshape(-1)
                    sc = (pred[..., k*4+3] * cls_sc).reshape(-1)
                    cl.append(torch.stack([cx, cy, r], dim=-1))
                    sl.append(sc)
                    ll.append(cls_idx.reshape(-1))

                circles_t = torch.cat(cl);  scores_t = torch.cat(sl);  labels_t = torch.cat(ll)
                keep = scores_t > conf_thr
                gt_circles = [batch['circles'][0][i].tolist()
                               for i in range(batch['circles'][0].shape[0])]
                raw.append((circles_t[keep].cpu(), scores_t[keep].cpu(),
                            labels_t[keep].cpu(), gt_circles))

        return raw, voc.idx2label

    finally:
        os.chdir(orig_dir)
        try: sys.path.remove(model_dir)
        except ValueError: pass
        _purge_project_modules()


def _build_pr_circle(raw_per_image, idx2label, nms_thr, iou_threshold):
    """Apply circle NMS at nms_thr, then build PR curve."""
    all_preds, all_gts = [], []
    labels_set = set(idx2label.values())
    for circles_t, scores_t, labels_t, gt_circles in raw_per_image:
        if circles_t.shape[0] > 0:
            mask = torch.zeros(circles_t.shape[0], dtype=torch.bool)
            for cid in torch.unique(labels_t):
                ci   = torch.where(labels_t == cid)[0]
                kept = _circle_nms(circles_t[ci], scores_t[ci], nms_thr)
                mask[ci[kept]] = True
            circles_t, scores_t, labels_t = circles_t[mask], scores_t[mask], labels_t[mask]

        pd_ = {v: [] for v in labels_set}
        for i in range(circles_t.shape[0]):
            lbl = idx2label[int(labels_t[i].item())]
            pd_[lbl].append(circles_t[i].tolist() + [scores_t[i].item()])
        all_preds.append(pd_)

        gd_ = {v: [] for v in labels_set}
        for gt in gt_circles:
            # GT circles have no label key stored per-circle; assign first label (single class)
            gd_[list(labels_set)[0]].append(gt)
        all_gts.append(gd_)

    return _build_pr(all_preds, all_gts, _circle_iou, iou_threshold)


# ── Baseline evaluation (single forward pass, fixed NMS) ─────────────────────

def eval_baseline_model(model_dir, config_path, ckpt_path, iou_threshold=0.5):
    orig_dir = os.getcwd()
    os.chdir(model_dir)
    _purge_project_modules()
    sys.path.insert(0, model_dir)
    try:
        from models.yolo import YOLOV1
        from dataset.voc import VOCDataset
        from torch.utils.data import DataLoader

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        dcfg, mcfg, tcfg = cfg['dataset_params'], cfg['model_params'], cfg['train_params']

        voc = VOCDataset('test', im_sets=dcfg['test_im_sets'], im_size=dcfg['im_size'],
                         S=mcfg['S'], B=mcfg['B'], C=dcfg['num_classes'])

        def _collate(batch):
            return {
                'image':  torch.stack([b['image'] for b in batch]),
                'bboxes': [b['bboxes'] for b in batch],
                'labels': [b['labels'] for b in batch],
            }

        loader = DataLoader(voc, batch_size=1, shuffle=False, collate_fn=_collate)
        model  = YOLOV1(im_size=dcfg['im_size'], num_classes=dcfg['num_classes'],
                        model_config=mcfg)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval().to(device)

        S, B, C     = mcfg['S'], mcfg['B'], dcfg['num_classes']
        conf_thr    = tcfg.get('eval_conf_threshold', 0.001)
        nms_thr     = tcfg.get('nms_threshold', 0.6)
        use_sigmoid = mcfg['use_sigmoid']

        all_preds, all_gts = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc='Baseline model (test set)'):
                im  = batch['image'].float().to(device)
                out = model(im)
                pred = out[0].clone() if out.dim() == 4 else out[0].reshape(S, S, 5*B+C).clone()
                aS   = pred.shape[0]

                if use_sigmoid:
                    pred[..., :5*B] = torch.sigmoid(pred[..., :5*B])
                pred = torch.clamp(pred, 0., 1.)

                cls_sc, cls_idx = torch.max(pred[..., 5*B:], dim=-1)
                sx = torch.arange(aS, dtype=torch.float32, device=device) / aS
                sy = torch.arange(aS, dtype=torch.float32, device=device) / aS
                sy, sx = torch.meshgrid(sy, sx, indexing='ij')

                bl, sl, ll = [], [], []
                for k in range(B):
                    cx = pred[..., k*5]   / aS + sx
                    cy = pred[..., k*5+1] / aS + sy
                    hw = 0.5 * torch.square(pred[..., k*5+2])
                    hh = 0.5 * torch.square(pred[..., k*5+3])
                    x1 = (cx - hw).reshape(-1); y1 = (cy - hh).reshape(-1)
                    x2 = (cx + hw).reshape(-1); y2 = (cy + hh).reshape(-1)
                    bl.append(torch.stack([x1, y1, x2, y2], dim=-1))
                    sl.append((pred[..., k*5+4] * cls_sc).reshape(-1))
                    ll.append(cls_idx.reshape(-1))

                boxes_t = torch.cat(bl); scores_t = torch.cat(sl); labels_t = torch.cat(ll)
                keep = scores_t > conf_thr
                boxes_t, scores_t, labels_t = boxes_t[keep], scores_t[keep], labels_t[keep]

                if boxes_t.shape[0] > 0:
                    mask = torch.zeros(boxes_t.shape[0], dtype=torch.bool, device=device)
                    for cid in torch.unique(labels_t):
                        ci   = torch.where(labels_t == cid)[0]
                        kept = torch.ops.torchvision.nms(boxes_t[ci], scores_t[ci], nms_thr)
                        mask[ci[kept]] = True
                    boxes_t, scores_t, labels_t = boxes_t[mask], scores_t[mask], labels_t[mask]

                pd_ = {v: [] for v in voc.idx2label.values()}
                for i in range(boxes_t.shape[0]):
                    lbl = voc.idx2label[int(labels_t[i].item())]
                    pd_[lbl].append(boxes_t[i].tolist() + [scores_t[i].item()])
                all_preds.append(pd_)

                gd_ = {v: [] for v in voc.idx2label.values()}
                gt_b = batch['bboxes'][0]; gt_l = batch['labels'][0]
                for i in range(gt_b.shape[0]):
                    lbl = voc.idx2label[int(gt_l[i].item())]
                    gd_[lbl].append(gt_b[i].tolist())
                all_gts.append(gd_)

        return _build_pr(all_preds, all_gts, _box_iou, iou_threshold)

    finally:
        os.chdir(orig_dir)
        try: sys.path.remove(model_dir)
        except ValueError: pass
        _purge_project_modules()


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_pr_curves(circle_06, circle_04, baseline_06, iou_threshold, output_path):
    all_labels = sorted(set(
        list(circle_06 or {}) + list(circle_04 or {}) + list(baseline_06 or {})
    ))
    n   = max(len(all_labels), 1)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle(f'Precision–Recall Curves  (test set, IoU@{iou_threshold:.2f})',
                 fontsize=13, fontweight='bold')

    curve_specs = [
        (baseline_06, 'Baseline rect (NMS=0.6)',  'steelblue', '-'),
        (circle_06,   'Circle (NMS=0.6)',          'tomato',    '--'),
        (circle_04,   'Circle (NMS=0.4, re-tuned)','seagreen',  '-'),
    ]

    for ax, label in zip(axes, all_labels):
        ax.set_title(f'Class: {label}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        for results, curve_label, color, ls in curve_specs:
            if results and label in results:
                rec, prec, ap = results[label]
                ax.plot(rec, prec, ls, color=color, linewidth=2,
                        label=f'{curve_label}  AP={ap:.3f}')
                ax.fill_between(rec, prec, alpha=0.07, color=color)

        ax.legend(loc='lower left', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PR curves: circle (NMS 0.4 & 0.6) vs baseline')
    parser.add_argument('--circle_ckpt',     default=CIRCLE_CKPT)
    parser.add_argument('--baseline_ckpt',   default=BASELINE_CKPT)
    parser.add_argument('--circle_dir',      default=CIRCLE_DIR)
    parser.add_argument('--baseline_dir',    default=BASELINE_DIR)
    parser.add_argument('--circle_config',   default='config/voc.yaml')
    parser.add_argument('--baseline_config', default='config/voc.yaml')
    parser.add_argument('--iou',             default=0.5, type=float)
    parser.add_argument('--output',          default=OUTPUT_PATH)
    args = parser.parse_args()

    circle_06 = circle_04 = baseline_06 = None

    # --- Circle model: one forward pass, two NMS settings ---
    if os.path.exists(args.circle_ckpt):
        print(f'Collecting circle model predictions from: {args.circle_ckpt}')
        raw, idx2label = _collect_circle_raw(
            args.circle_dir,
            os.path.join(args.circle_dir, args.circle_config),
            args.circle_ckpt,
        )
        circle_06 = _build_pr_circle(raw, idx2label, nms_thr=0.6, iou_threshold=args.iou)
        circle_04 = _build_pr_circle(raw, idx2label, nms_thr=0.4, iou_threshold=args.iou)
        for lbl in circle_06:
            print(f'  Circle NMS=0.6  AP@{args.iou:.2f} [{lbl}] = {circle_06[lbl][2]:.4f}')
            print(f'  Circle NMS=0.4  AP@{args.iou:.2f} [{lbl}] = {circle_04[lbl][2]:.4f}')
    else:
        print(f'Circle checkpoint not found: {args.circle_ckpt}')

    # --- Baseline model ---
    if os.path.exists(args.baseline_ckpt):
        print(f'\nEvaluating baseline model from: {args.baseline_ckpt}')
        baseline_06 = eval_baseline_model(
            args.baseline_dir,
            os.path.join(args.baseline_dir, args.baseline_config),
            args.baseline_ckpt,
            iou_threshold=args.iou,
        )
        for lbl, (_, _, ap) in baseline_06.items():
            print(f'  Baseline NMS=0.6  AP@{args.iou:.2f} [{lbl}] = {ap:.4f}')
    else:
        print(f'Baseline checkpoint not found: {args.baseline_ckpt}')

    if any(r is not None for r in [circle_06, circle_04, baseline_06]):
        plot_pr_curves(circle_06, circle_04, baseline_06, args.iou, args.output)
    else:
        print('No results to plot.')
