"""
Plot training curves for circle and baseline YOLOv1 models.

Usage:
    python plot_curves.py
    """
import argparse
import json
import os

import matplotlib.pyplot as plt

_HERE        = os.path.dirname(os.path.abspath(__file__))
CIRCLE_LOG   = os.path.join(_HERE, 'Yolov1-PyTorch',          'results', 'training_log_circle.json')
BASELINE_LOG = os.path.join(_HERE, 'Yolov1-PyTorch Baseline', 'results', 'training_log_baseline.json')
OUTPUT_PATH  = os.path.join(_HERE, 'training_curves.png')


def load_log(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_curves(circle_log, baseline_log, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('YOLOv1 Training Curves', fontsize=14, fontweight='bold')

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_title('Loss over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    if circle_log:
        epochs = [r['epoch'] for r in circle_log]
        ax.plot(epochs, [r['train_loss'] for r in circle_log],
                '-',  color='tab:blue',   label='Circle – Train')
        ax.plot(epochs, [r['val_loss'] for r in circle_log],
                '--', color='tab:blue',   label='Circle – Val')

    if baseline_log:
        epochs = [r['epoch'] for r in baseline_log]
        ax.plot(epochs, [r['train_loss'] for r in baseline_log],
                '-',  color='tab:orange', label='Baseline – Train')
        ax.plot(epochs, [r['val_loss'] for r in baseline_log],
                '--', color='tab:orange', label='Baseline – Val')

    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── mAP ───────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_title('Validation mAP@0.5 over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP@0.5')
    has_map = False

    if circle_log:
        pts = [(r['epoch'], r['val_map50']) for r in circle_log if 'val_map50' in r]
        if pts:
            ep, mp = zip(*pts)
            ax.plot(ep, mp, '-o', color='tab:blue', markersize=4, label='Circle')
            has_map = True

    if baseline_log:
        pts = [(r['epoch'], r['val_map50']) for r in baseline_log if 'val_map50' in r]
        if pts:
            ep, mp = zip(*pts)
            ax.plot(ep, mp, '-o', color='tab:orange', markersize=4, label='Baseline')
            has_map = True

    if has_map:
        ax.legend()
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5,
                'No mAP data yet.\nSet eval_map_interval > 0 in voc.yaml and retrain.',
                ha='center', va='center', transform=ax.transAxes, color='gray',
                fontsize=10)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved plot to: {output_path}')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot YOLOv1 training curves')
    parser.add_argument('--circle_log',   default=CIRCLE_LOG,   help='Path to circle model JSON log')
    parser.add_argument('--baseline_log', default=BASELINE_LOG, help='Path to baseline model JSON log')
    parser.add_argument('--output',       default=OUTPUT_PATH,  help='Output PNG path')
    args = parser.parse_args()

    circle_log   = load_log(args.circle_log)
    baseline_log = load_log(args.baseline_log)

    if circle_log is None and baseline_log is None:
        print('No log files found. Train the models first (logs are saved to the task_name directory).')
    else:
        if circle_log:
            print(f'Circle model:   {len(circle_log)} epoch(s) — {args.circle_log}')
        else:
            print(f'Circle model:   log not found ({args.circle_log})')
        if baseline_log:
            print(f'Baseline model: {len(baseline_log)} epoch(s) — {args.baseline_log}')
        else:
            print(f'Baseline model: log not found ({args.baseline_log})')

        plot_curves(circle_log, baseline_log, args.output)
