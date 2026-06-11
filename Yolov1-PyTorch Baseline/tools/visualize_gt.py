"""
Saves ground-truth bounding-box overlays for the test-set indices listed in
train_params.infer_sample_indices (config/voc.yaml).

Run from the Yolov1-PyTorch Baseline directory:
    python -m tools.visualize_gt --config config/voc.yaml
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import yaml

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def tensor_to_bgr(im_tensor):
    """Reverse ImageNet normalisation and return an OpenCV BGR uint8 image."""
    img = im_tensor.permute(1, 2, 0).numpy()   # H W C, float32
    img = (img * STD + MEAN).clip(0, 1)
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/voc.yaml', type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg['dataset_params']
    mcfg = cfg['model_params']
    tcfg = cfg['train_params']

    sys.path.insert(0, os.getcwd())
    from dataset.voc import VOCDataset

    voc = VOCDataset('test',
                     im_sets=dcfg['test_im_sets'],
                     im_size=dcfg['im_size'],
                     S=mcfg['S'], B=mcfg['B'], C=dcfg['num_classes'])

    sample_indices = tcfg.get('infer_sample_indices', list(range(10)))

    out_dir = os.path.join('samples', 'gt')
    os.makedirs(out_dir, exist_ok=True)

    for dataset_idx in sample_indices:
        if dataset_idx >= len(voc):
            print(f"Index {dataset_idx} out of range (dataset size {len(voc)}), skipping.")
            continue

        sample = voc[dataset_idx]
        im_tensor  = sample['image']           # C H W, normalised
        bboxes     = sample['bboxes']          # N x 4, normalised 0-1 (x1 y1 x2 y2)
        labels     = sample['labels']          # N

        img = tensor_to_bgr(im_tensor)
        H, W = img.shape[:2]

        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i].tolist()
            x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
            label_name = voc.idx2label[int(labels[i].item())]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label_name, (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)

        fname = f'gt_{dataset_idx:04d}.png'
        cv2.imwrite(os.path.join(out_dir, fname), img)
        print(f"Saved {fname}  ({len(bboxes)} GT boxes)")

    print(f"\nDone. Images saved to {out_dir}/")


if __name__ == '__main__':
    main()
