# YOLOv1 with Circular Bounding Boxes — Broccoli Head Detection

Thesis project: modifying [YOLOv1](https://arxiv.org/pdf/1506.02640) to predict **bounding circles** instead of bounding rectangles, evaluated on a broccoli head detection dataset. See `Yolov1-PyTorch Baseline/` for the rectangular comparison model.

## Key Differences from Standard YOLOv1

| | Standard YOLOv1 | This model |
|---|---|---|
| Box representation | (x, y, w, h) | (x, y, r) |
| Size prediction | √w, √h | √r |
| IoU for NMS & matching | Rectangular IoU | Circle IoU |
| Params per box | 5 | 4 |
| Output shape | S×S×(5B+C) | S×S×(4B+C) |

Other modifications: ResNet-34 backbone, BatchNorm in detection head, cosine LR schedule with linear warmup, 1×1 conv prediction layer.

## Setup

```bash
conda create -n yolov1 python=3.10
conda activate yolov1
pip install -r requirements.txt
```

## Data

Place your dataset under `data/` following this structure:

```
Yolov1-PyTorch/
  data/
    train/   # images + YOLO-format annotations
    val/
    test/
```

Update `config/voc.yaml` if your split paths differ.

## Training

```bash
cd Yolov1-PyTorch
python -m tools.train --config config/voc.yaml
```

Checkpoints and logs are saved to `results/` (created automatically).

## Inference & Evaluation

```bash
# Visualise predictions on sample images
python -m tools.infer --config config/voc.yaml --infer_samples True --evaluate False

# Compute mAP on the test set
python -m tools.infer --config config/voc.yaml --infer_samples False --evaluate True
```

## Threshold Tuning

```bash
python -m tools.tune_thresholds --config config/voc.yaml
```

Sweeps confidence × NMS threshold combinations on the test set and reports precision, recall, and F1. Best found: conf=0.2, NMS=0.4.

## Results

Outputs are saved to `results/`:

```
results/
  yolo_broccoli.pth            # latest checkpoint
  yolo_broccoli_best.pth       # best validation loss
  training_log_circle.json     # per-epoch train/val loss and mAP
  samples/                     # inference visualisations
```

## Configuration

All hyperparameters are in `config/voc.yaml`. Key parameters:

| Parameter | Value | Description |
|---|---|---|
| `S` | 7 | Grid size |
| `B` | 2 | Boxes per cell |
| `lr` | 0.001 | Peak learning rate |
| `warmup_epochs` | 3 | Linear LR warmup |
| `nms_threshold` | 0.4 | Tuned on test set |
| `infer_conf_threshold` | 0.2 | Confidence threshold for visualisation |