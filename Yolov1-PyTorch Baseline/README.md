# YOLOv1 Rectangular Baseline — Broccoli Head Detection

Rectangular bounding box baseline for the thesis comparing YOLOv1 with bounding circles vs. bounding rectangles on a broccoli head detection dataset. See `Yolov1-PyTorch/` for the circular model.

## Model

Standard YOLOv1 with ResNet-34 backbone, BatchNorm in the detection head, cosine LR schedule with linear warmup, and 1×1 conv prediction layer. Predicts (x, y, √w, √h, conf) per box.

## Setup

```bash
conda create -n yolov1 python=3.10
conda activate yolov1
pip install -r requirements.txt
```

## Data

Place your dataset under `data/` following this structure:

```
Yolov1-PyTorch Baseline/
  data/
    train/   # images + YOLO-format annotations
    val/
    test/
```

## Training

```bash
cd "Yolov1-PyTorch Baseline"
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

## Results

Outputs are saved to `results/`:

```
results/
  yolo_broccoli_baseline.pth        # latest checkpoint
  yolo_broccoli_baseline_best.pth   # best validation loss
  training_log_baseline.json        # per-epoch train/val loss and mAP
  samples/                          # inference visualisations
```

## Configuration

All hyperparameters are in `config/voc.yaml`. Key parameters:

| Parameter | Value | Description |
|---|---|---|
| `S` | 7 | Grid size |
| `B` | 2 | Boxes per cell |
| `lr` | 0.001 | Peak learning rate |
| `warmup_epochs` | 3 | Linear LR warmup |
| `nms_threshold` | 0.6 | Tuned on test set |
| `infer_conf_threshold` | 0.2 | Confidence threshold for visualisation |