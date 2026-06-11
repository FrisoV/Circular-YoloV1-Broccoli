import math
import torch

def circle_iou(circles1, circles2, eps: float = 1e-7):
    """
    circles1: Tensor[N, 3] with (x, y, r)
    circles2: Tensor[M, 3] with (x, y, r)

    Returns:
        iou:   Tensor[N, M]
        union: Tensor[N, M]
    """
    if circles1.numel() == 0 or circles2.numel() == 0:
        return (circles1.new_zeros((circles1.shape[0], circles2.shape[0])),
                circles1.new_zeros((circles1.shape[0], circles2.shape[0])))

    c1 = circles1[:, :2]          # [N, 2]
    c2 = circles2[:, :2]          # [M, 2]
    r1 = torch.clamp(circles1[:, 2].unsqueeze(1), min=0.0)  # [N, 1]
    r2 = torch.clamp(circles2[:, 2].unsqueeze(0), min=0.0)  # [1, M]
    r1_safe = torch.clamp(r1, min=eps)
    r2_safe = torch.clamp(r2, min=eps)

    # Pairwise center distances
    dx = c1[:, None, 0] - c2[None, :, 0]    # [N, M]
    dy = c1[:, None, 1] - c2[None, :, 1]    # [N, M]
    d2 = dx * dx + dy * dy                  # [N, M]
    d = torch.sqrt(d2)                      # true distance
    d_safe = torch.clamp(d, min=eps)        # for divisions

    # Areas
    area1 = math.pi * r1 * r1               # [N, 1]
    area2 = math.pi * r2 * r2               # [1, M]

    # Initialize intersection
    inter = torch.zeros_like(d)

    # Case 1: no overlap
    no_overlap = d >= (r1 + r2)
    # (inter remains 0 there)

    # Case 2: one circle fully inside the other (including identical circles)
    contain = d <= torch.abs(r1 - r2)
    inter_contain = torch.min(area1, area2)  # area of smaller circle
    inter = torch.where(contain, inter_contain, inter)

    # Case 3: partial overlap
    mask = (~no_overlap) & (~contain)
    if mask.any():
        d_m = d_safe[mask]
        r1_m = r1.expand_as(d)[mask]
        r2_m = r2.expand_as(d)[mask]

        # Segment angles (clamped for numerical stability)
        cos1 = (d_m**2 + r1_m**2 - r2_m**2) / (2 * d_m * r1_m)
        cos2 = (d_m**2 + r2_m**2 - r1_m**2) / (2 * d_m * r2_m)
        cos1 = torch.nan_to_num(torch.clamp(cos1, -1.0, 1.0), nan=1.0, posinf=1.0, neginf=-1.0)
        cos2 = torch.nan_to_num(torch.clamp(cos2, -1.0, 1.0), nan=1.0, posinf=1.0, neginf=-1.0)

        part1 = r1_m**2 * torch.acos(cos1)
        part2 = r2_m**2 * torch.acos(cos2)
        part3 = 0.5 * torch.sqrt(
            torch.clamp(
                (-d_m + r1_m + r2_m) *
                ( d_m + r1_m - r2_m) *
                ( d_m - r1_m + r2_m) *
                ( d_m + r1_m + r2_m),
                min=0.0,
            )
        )

        inter_m = torch.nan_to_num(part1 + part2 - part3, nan=0.0, posinf=0.0, neginf=0.0) # Final formula for intersection area of two circles in case 3
        inter[mask] = inter_m

    # Union and IoU
    union = area1 + area2 - inter
    union = torch.clamp(union, min=eps)
    iou = torch.nan_to_num(inter / union, nan=0.0, posinf=0.0, neginf=0.0)

    return iou, union

def boxes_to_circles(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert DETR-style boxes [cx, cy, w, h] to circles [cx, cy, r].

    All coordinates are normalized to [0, 1] relative to image width/height.

    Returns:
        Tensor of shape (..., 3) in [cx, cy, r].
    """
    if boxes.numel() == 0:
        # empty (no predictions)
        return boxes.new_zeros(*boxes.shape[:-1], 3)

    cx, cy, w, h = boxes.unbind(-1)

    r = 0.5 * torch.min(w, h)
    return torch.stack((cx, cy, r), dim=-1)
