import os
import albumentations as albu
import cv2
import torch
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
import os
import json
import math


def load_images_and_anns(im_sets, label2idx, ann_fname, split):
    """
    Custom loader for LabelMe-style PNG + JSON annotations.
    Expects:
      data/train/*.png + *.json
      data/val/*.png + *.json
      data/test/*.png + *.json
    """

    im_infos = []

    for im_set in im_sets:
        for fname in os.listdir(im_set):
            if not fname.endswith(".png"):
                continue

            img_path = os.path.join(im_set, fname)
            json_path = img_path.replace(".png", ".json")

            if not os.path.exists(json_path):
                continue

            with open(json_path, "r") as f:
                data = json.load(f)

            width = data["imageWidth"]
            height = data["imageHeight"]

            detections = []
            any_valid_object = False

            for shape in data["shapes"]:
                if shape["shape_type"] != "circle":
                    continue

                label_name = shape["label"]
                if label_name not in label2idx:
                    continue

                (cx, cy), (px, py) = shape["points"]
                radius = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

                xmin = int(max(cx - radius, 0))
                ymin = int(max(cy - radius, 0))
                xmax = int(min(cx + radius, width))
                ymax = int(min(cy + radius, height))

                det = {
                    "label": label2idx[label_name],
                    "bbox": [xmin, ymin, xmax, ymax],
                }

                detections.append(det)
                any_valid_object = True

            if any_valid_object:
                im_infos.append({
                    "img_id": os.path.splitext(fname)[0],
                    "filename": img_path,
                    "width": width,
                    "height": height,
                    "detections": detections
                })

    print(f"Total {len(im_infos)} images found for split={split}")
    return im_infos



class VOCDataset(Dataset):
    def __init__(self, split, im_sets, im_size=448, S=7, B=2, C=1): # C was for 20 classes in VOC, now we have just 1 class
        self.split = split
        # Imagesets for this dataset instance (data/test)
        self.im_sets = im_sets
        self.fname = 'trainval' if self.split == 'train' else 'test'
        self.im_size = im_size
        # Grid size, B and C parameter for target setting
        self.S = S
        self.B = B
        self.C = C

        # Train and test transformations
        self.transforms = {
            'train': albu.Compose([
                albu.HorizontalFlip(p=0.5),
                albu.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=(-0.2, 0.2),
                    always_apply=True
                ),
                albu.ColorJitter(
                    brightness=(0.8, 1.2),
                    contrast=(0.8, 1.2),
                    saturation=(0.8, 1.2),
                    hue=(-0.2, 0.2),
                    always_apply=None,
                    p=0.5,
                ),
                albu.Resize(self.im_size, self.im_size)], # Automatically resizes images
                bbox_params=albu.BboxParams(format='pascal_voc',
                                            label_fields=['labels'])),
            'test': albu.Compose([
                albu.Resize(self.im_size, self.im_size),
                ],
                bbox_params=albu.BboxParams(format='pascal_voc',
                                            label_fields=['labels']))
            ,
            'val': albu.Compose([
                albu.Resize(self.im_size, self.im_size),
                ],
                bbox_params=albu.BboxParams(format='pascal_voc',
                                            label_fields=['labels']))
        }

        classes = [
            'broccoli'
        ]
        classes = sorted(classes)
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        print(self.idx2label)
        self.im_info = load_images_and_anns(self.im_sets,
                                                self.label2idx,
                                                self.fname,
                                                self.split)
    
    def __len__(self):
        return len(self.im_info)
    
    def __getitem__(self, index):
        im_info = self.im_info[index]

        # Load Image
        im = cv2.imread(im_info['filename'])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # Get annotations for this image
        bboxes = [detection['bbox'] for detection in im_info['detections']]
        labels = [detection['label'] for detection in im_info['detections']]

        # Transform Image and ann according to augmentations list
        transformed_info = self.transforms[self.split](image=im,
                                                       bboxes=bboxes,
                                                       labels=labels)
        im = transformed_info['image']
        bboxes = torch.as_tensor(transformed_info['bboxes'])
        labels = torch.as_tensor(transformed_info['labels'])

        # Convert image to tensor and normalize
        im_tensor = torch.from_numpy(im / 255.).permute((2, 0, 1)).float()
        im_tensor_channel_0 = (torch.unsqueeze(im_tensor[0], 0) - 0.485) / 0.229
        im_tensor_channel_1 = (torch.unsqueeze(im_tensor[1], 0) - 0.456) / 0.224
        im_tensor_channel_2 = (torch.unsqueeze(im_tensor[2], 0) - 0.406) / 0.225
        im_tensor = torch.cat((im_tensor_channel_0,
                               im_tensor_channel_1,
                               im_tensor_channel_2), 0)
        bboxes_tensor = torch.as_tensor(bboxes)
        labels_tensor = torch.as_tensor(labels)

        # Build Target for Yolo
        target_dim = 5 * self.B + self.C
        h, w = im.shape[:2]
        yolo_targets = torch.zeros(self.S, self.S, target_dim)

        # Height and width of grid cells is H // S
        cell_pixels = h // self.S

        if len(bboxes) > 0:
            # Convert x1y1x2y2 to xywh format
            box_widths = bboxes_tensor[:, 2] - bboxes_tensor[:, 0]
            box_heights = bboxes_tensor[:, 3] - bboxes_tensor[:, 1]
            box_center_x = bboxes_tensor[:, 0] + 0.5 * box_widths
            box_center_y = bboxes_tensor[:, 1] + 0.5 * box_heights

            # Get cell i,j from xc, yc
            box_i = torch.floor(box_center_x / cell_pixels).long()
            box_j = torch.floor(box_center_y / cell_pixels).long()

            # xc offset from cell topleft
            box_xc_cell_offset = (box_center_x - box_i*cell_pixels) / cell_pixels
            box_yc_cell_offset = (box_center_y - box_j*cell_pixels) / cell_pixels

            # w, h targets normalized to 0-1
            box_w_label = box_widths / w
            box_h_label = box_heights / h

            # Update the target array for all bboxes
            for idx, b in enumerate(range(bboxes_tensor.size(0))):
                # Make target of the exact same shape as prediction
                for k in range(self.B):
                    s = 5 * k
                    # target_ij = [xc_offset,yc_offset,sqrt(w),sqrt(h), conf, cls_label]
                    yolo_targets[box_j[idx], box_i[idx], s] = box_xc_cell_offset[idx]
                    yolo_targets[box_j[idx], box_i[idx], s+1] = box_yc_cell_offset[idx]
                    yolo_targets[box_j[idx], box_i[idx], s+2] = box_w_label[idx].sqrt()
                    yolo_targets[box_j[idx], box_i[idx], s+3] = box_h_label[idx].sqrt()
                    yolo_targets[box_j[idx], box_i[idx], s+4] = 1.0
                label = int(labels[b])
                cls_target = torch.zeros((self.C,))
                cls_target[label] = 1.
                yolo_targets[box_j[idx], box_i[idx], 5 * self.B:] = cls_target
        # For training, we use yolo_targets(xoffset, yoffset, sqrt(w), sqrt(h))
        # For evaluation we use bboxes_tensor (x1, y1, x2, y2)
        # Below we normalize bboxes tensor to be between 0-1
        # as thats what evaluation script expects so (x1/w, y1/h, x2/w, y2/h)
        if len(bboxes) > 0:
            bboxes_tensor /= torch.Tensor([[w, h, w, h]]).expand_as(bboxes_tensor)
        targets = {
            'image': im_tensor,
            'bboxes': bboxes_tensor,
            'labels': labels_tensor,
            'yolo_targets': yolo_targets,
            'file_path': im_info['filename']
        }
        return targets
