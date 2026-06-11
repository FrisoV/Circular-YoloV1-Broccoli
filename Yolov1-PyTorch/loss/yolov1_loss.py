import torch
import torch.nn as nn
import math
import sys
from pathlib import Path

# Add parent directory to path to import circle_ops
sys.path.insert(0, str(Path(__file__).parent.parent))
from circle_ops.circle_ops import circle_iou


def get_iou(boxes1, boxes2):
    r"""
    IOU between two sets of boxes
    """
    # Area of boxes (x2-x1)*(y2-y1)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # Get top left x1,y1 coordinate
    x_left = torch.max(boxes1[..., 0], boxes2[..., 0])
    y_top = torch.max(boxes1[..., 1], boxes2[..., 1])

    # Get bottom right x2,y2 coordinate
    x_right = torch.min(boxes1[..., 2], boxes2[..., 2])
    y_bottom = torch.min(boxes1[..., 3], boxes2[..., 3])

    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    union = area1.clamp(min=0) + area2.clamp(min=0) - intersection_area
    iou = intersection_area / (union + 1E-6)
    return iou


def get_circle_iou_batch(circles1, circles2):
    r"""
    Compute circle IoU for batched tensors.
    circles1: (N, 3) with (cx, cy, r)
    circles2: (M, 3) with (cx, cy, r)
    Returns: (N, M) IoU matrix
    """
    iou, _ = circle_iou(circles1, circles2)
    return iou


def get_elementwise_circle_iou(circles1, circles2):
    r"""
    Compute IoU for matching circle tensors with identical leading shape.
    The last dimension must be (cx, cy, r).
    """
    flat1 = circles1.reshape(-1, 3)
    flat2 = circles2.reshape(-1, 3)
    iou, _ = circle_iou(flat1, flat2)
    return torch.diagonal(iou, dim1=0, dim2=1).reshape(circles1.shape[:-1])


class YOLOV1Loss(nn.Module):
    r"""
    Loss module for YoloV1 which caters to the following components:
    1. Localization Loss for responsible predictor circles
    2. Objectness Loss for responsible predictor circles
    3. Objectness Loss for non-responsible predictor circles of cells assigned with objects
    4. Objectness Loss for ALL predictor circles of cells not assigned with objects
    5. Classification Loss
    """
    def __init__(self, S=7, B=2, C=20, use_circles=False):
        super(YOLOV1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.use_circles = use_circles
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, preds, targets, use_sigmoid=False):
        r"""
        Main method of loss computation
        :param preds: (Batch, S*S*(4B+C) for circles or 5B+C for boxes) tensor
        :param targets: (Batch, S, S, (4B+C) for circles or 5B+C for boxes) tensor
        :param use_sigmoid: Whether to use sigmoid activation for circle/box predictions
        """
        batch_size = preds.size(0)
        
        if self.use_circles:
            # For circles: [cx_offset, cy_offset, sqrt(r), conf]
            num_params_per_box = 4
            output_channels = 4 * self.B + self.C
        else:
            # For boxes: [cx_offset, cy_offset, sqrt(w), sqrt(h), conf]
            num_params_per_box = 5
            output_channels = 5 * self.B + self.C

        # preds -> (Batch, S, S, 4B+C or 5B+C)
        preds = preds.reshape(batch_size, self.S, self.S, output_channels)

        # Generally sigmoid leads to quicker convergence
        if use_sigmoid:
            preds[..., :num_params_per_box * self.B] = torch.nn.functional.sigmoid(
                preds[..., :num_params_per_box * self.B]
            )

        # Shifts for all grid cell locations
        shifts_x = torch.arange(0, self.S,
                                dtype=torch.float32,
                                device=preds.device) * 1 / float(self.S)
        shifts_y = torch.arange(0, self.S,
                                dtype=torch.float32,
                                device=preds.device) * 1 / float(self.S)

        # Create a grid using these shifts
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        # shifts -> (1, S, S, B)
        shifts_x = shifts_x.reshape((1, self.S, self.S, 1)).repeat(1, 1, 1, self.B)
        shifts_y = shifts_y.reshape((1, self.S, self.S, 1)).repeat(1, 1, 1, self.B)

        # pred_boxes -> (Batch_size, S, S, B, params_per_box)
        pred_boxes = preds[..., :num_params_per_box * self.B].reshape(
            batch_size,
            self.S,
            self.S,
            self.B,
            num_params_per_box,
        )

        if self.use_circles:
            pred_boxes_x = (pred_boxes[..., 0] / self.S + shifts_x)[..., None]
            pred_boxes_y = (pred_boxes[..., 1] / self.S + shifts_y)[..., None]
            pred_boxes_r = torch.square(pred_boxes[..., 2])[..., None]
            pred_boxes_geom = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_r], dim=-1)

            # target_boxes -> (Batch_size, S, S, B, 4)
            target_boxes = targets[..., :num_params_per_box * self.B].reshape(
                batch_size,
                self.S,
                self.S,
                self.B,
                num_params_per_box,
            )
            target_boxes_x = (target_boxes[..., 0] / self.S + shifts_x)[..., None]
            target_boxes_y = (target_boxes[..., 1] / self.S + shifts_y)[..., None]
            target_boxes_r = torch.square(target_boxes[..., 2])[..., None]
            target_boxes_geom = torch.cat([target_boxes_x, target_boxes_y, target_boxes_r], dim=-1)

            # iou -> (Batch_size, S, S, B)
            iou = get_elementwise_circle_iou(pred_boxes_geom, target_boxes_geom)
        else:
            # xc_offset yc_offset w h -> x1 y1 x2 y2 (normalized 0-1)
            # x_center = (xc_offset / S + shift_x)
            # x1 = x_center - 0.5 * w
            # x2 = x_center + 0.5 * w
            pred_boxes_x1 = ((pred_boxes[..., 0]/self.S + shifts_x)
                             - 0.5*torch.square(pred_boxes[..., 2]))
            pred_boxes_x1 = pred_boxes_x1[..., None]
            pred_boxes_y1 = ((pred_boxes[..., 1]/self.S + shifts_y)
                             - 0.5*torch.square(pred_boxes[..., 3]))
            pred_boxes_y1 = pred_boxes_y1[..., None]
            pred_boxes_x2 = ((pred_boxes[..., 0]/self.S + shifts_x)
                             + 0.5*torch.square(pred_boxes[..., 2]))
            pred_boxes_x2 = pred_boxes_x2[..., None]
            pred_boxes_y2 = ((pred_boxes[..., 1]/self.S + shifts_y)
                             + 0.5*torch.square(pred_boxes[..., 3]))
            pred_boxes_y2 = pred_boxes_y2[..., None]
            pred_boxes_x1y1x2y2 = torch.cat([
                pred_boxes_x1,
                pred_boxes_y1,
                pred_boxes_x2,
                pred_boxes_y2], dim=-1)

            # target_boxes -> (Batch_size, S, S, B, 5)
            target_boxes = targets[..., :num_params_per_box * self.B].reshape(
                batch_size,
                self.S,
                self.S,
                self.B,
                num_params_per_box,
            )
            target_boxes_x1 = ((target_boxes[..., 0] / self.S + shifts_x)
                               - 0.5 * torch.square(target_boxes[..., 2]))
            target_boxes_x1 = target_boxes_x1[..., None]
            target_boxes_y1 = ((target_boxes[..., 1] / self.S + shifts_y)
                               - 0.5 * torch.square(target_boxes[..., 3]))
            target_boxes_y1 = target_boxes_y1[..., None]
            target_boxes_x2 = ((target_boxes[..., 0] / self.S + shifts_x)
                               + 0.5 * torch.square(target_boxes[..., 2]))
            target_boxes_x2 = target_boxes_x2[..., None]
            target_boxes_y2 = ((target_boxes[..., 1] / self.S + shifts_y)
                               + 0.5 * torch.square(target_boxes[..., 3]))
            target_boxes_y2 = target_boxes_y2[..., None]
            target_boxes_x1y1x2y2 = torch.cat([
                target_boxes_x1,
                target_boxes_y1,
                target_boxes_x2,
                target_boxes_y2
            ], dim=-1)

            # iou -> (Batch_size, S, S, B)
            iou = get_iou(pred_boxes_x1y1x2y2, target_boxes_x1y1x2y2)

        # max_iou_val/max_iou_idx -> (Batch_size, S, S, 1)
        max_iou_val, max_iou_idx = iou.max(dim=-1, keepdim=True)
        # IoU is used as a regression target for confidence; do not backprop through it.
        max_iou_val = max_iou_val.detach()

        #########################
        # Indicator Definitions #
        #########################
        # before max_iou_idx -> (Batch_size, S, S, 1) Eg [[0], [1], [0], [0]]
        # after repeating max_iou_idx -> (Batch_size, S, S, B)
        # Eg. [[0, 0], [1, 1], [0, 0], [0, 0]] assuming B = 2
        max_iou_idx = max_iou_idx.repeat(1, 1, 1, self.B)
        # bb_idxs -> (Batch_size, S, S, B)
        #  Eg. [[0, 1], [0, 1], [0, 1], [0, 1]] assuming B = 2
        bb_idxs = (torch.arange(self.B).reshape(1, 1, 1, self.B).expand_as(max_iou_idx)
                   .to(preds.device))
        # is_max_iou_box -> (Batch_size, S, S, B)
        # Eg. [[True, False], [False, True], [True, False], [True, False]]
        # only the index which is max iou boxes index will be 1 rest all 0
        is_max_iou_box = (max_iou_idx == bb_idxs).long()

        # obj_indicator -> (Batch_size, S, S, 1)
        obj_indicator = targets[..., 3:4] if self.use_circles else targets[..., 4:5]

        # Loss definitions start from here

        #######################
        # Classification Loss #
        #######################
        cls_target = targets[..., num_params_per_box * self.B:]
        cls_preds = preds[..., num_params_per_box * self.B:]
        cls_mse = (cls_preds - cls_target) ** 2
        # Only keep losses from cells with object assigned
        cls_mse = (obj_indicator * cls_mse).sum()

        ######################################################
        # Objectness Loss (For responsible predictor boxes ) #
        ######################################################
        # indicator is now object_cells * is_best_box
        is_max_box_obj_indicator = is_max_iou_box * obj_indicator
        obj_mse = (pred_boxes[..., 3] - max_iou_val) ** 2 if self.use_circles else (pred_boxes[..., 4] - max_iou_val) ** 2
        # Only keep losses from boxes of cells with object assigned
        # and that box which is the responsible predictor
        obj_mse = (is_max_box_obj_indicator * obj_mse).sum()

        #####################
        # Localization Loss #
        #####################
        x_mse = (pred_boxes[..., 0] - target_boxes[..., 0]) ** 2
        # Only keep losses from boxes of cells with object assigned
        # and that box which is the responsible predictor
        x_mse = (is_max_box_obj_indicator * x_mse).sum()

        y_mse = (pred_boxes[..., 1] - target_boxes[..., 1]) ** 2
        y_mse = (is_max_box_obj_indicator * y_mse).sum()
        if self.use_circles:
            r_sqrt_mse = (pred_boxes[..., 2] - target_boxes[..., 2]) ** 2
            r_sqrt_mse = (is_max_box_obj_indicator * r_sqrt_mse).sum()
        else:
            w_sqrt_mse = (pred_boxes[..., 2] - target_boxes[..., 2]) ** 2
            w_sqrt_mse = (is_max_box_obj_indicator * w_sqrt_mse).sum()
            h_sqrt_mse = (pred_boxes[..., 3] - target_boxes[..., 3]) ** 2
            h_sqrt_mse = (is_max_box_obj_indicator * h_sqrt_mse).sum()

        #################################################
        # Objectness Loss
        # For boxes of cells assigned with object that
        # aren't responsible predictor boxes
        # and for boxes of cell not assigned with object
        #################################################
        no_object_indicator = 1 - is_max_box_obj_indicator
        no_obj_mse = (pred_boxes[..., 3] - torch.zeros_like(pred_boxes[..., 3])) ** 2 if self.use_circles else (pred_boxes[..., 4] - torch.zeros_like(pred_boxes[..., 4])) ** 2
        no_obj_mse = (no_object_indicator * no_obj_mse).sum()

        ##############
        # Total Loss #
        ##############
        if self.use_circles:
            loss = self.lambda_coord * (x_mse + y_mse + r_sqrt_mse)
        else:
            loss = self.lambda_coord*(x_mse + y_mse + w_sqrt_mse + h_sqrt_mse)
        loss += cls_mse + obj_mse
        loss += self.lambda_noobj*no_obj_mse
        loss = loss / batch_size
        return loss
