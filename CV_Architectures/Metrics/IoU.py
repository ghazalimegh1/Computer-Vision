def iou(ground_truth, prediction):
    xg_min, yg_min, xg_max, yg_max = ground_truth
    xp_min, yp_min, xp_max, yp_max = prediction

    # intersection coordinates
    xi_min, yi_min = max(xg_min, xp_min), max(yg_min, yp_min)
    xi_max, yi_max = min(xg_max, xp_max), min(yg_max, yp_max)

    intersection_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
    ground_truth_area = (xg_max - xg_min) * (yg_max - yg_min)
    prediction_area = (xp_max - xp_min) * (yp_max - yp_min)

    union_area = ground_truth_area + prediction_area - intersection_area

    if union_area == 0:
        return 0.0
    iou = intersection_area / union_area
    return iou



import torch


def giou_loss(pred_boxes, gt_boxes, eps=1e-7):
    """
    Computes GIoU Loss between predicted and ground-truth bounding boxes.

    Args:
        pred_boxes (Tensor): Predicted boxes of shape (N, 4) in [x1, y1, x2, y2] format.
        gt_boxes (Tensor): Ground-truth boxes of shape (N, 4) in [x1, y1, x2, y2] format.
        eps (float): Small value to avoid division by zero.

    Returns:
        Tensor: Mean GIoU loss over the batch.
    """
    # 1. Unpack coordinates
    p_x1, p_y1, p_x2, p_y2 = (
        pred_boxes[:, 0],
        pred_boxes[:, 1],
        pred_boxes[:, 2],
        pred_boxes[:, 3],
    )
    g_x1, g_y1, g_x2, g_y2 = (
        gt_boxes[:, 0],
        gt_boxes[:, 1],
        gt_boxes[:, 2],
        gt_boxes[:, 3],
    )

    # 2. Compute Intersection Area
    inter_x1 = torch.max(p_x1, g_x1)
    inter_y1 = torch.max(p_y1, g_y1)
    inter_x2 = torch.min(p_x2, g_x2)
    inter_y2 = torch.min(p_y2, g_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # 3. Compute Individual Areas and Union Area
    pred_area = (p_x2 - p_x1) * (p_y2 - p_y1)
    gt_area = (g_x2 - g_x1) * (g_y2 - g_y1)
    union_area = pred_area + gt_area - inter_area + eps

    # 4. Compute Standard IoU
    iou = inter_area / union_area

    # 5. Compute Smallest Enclosing Box (C) coordinates
    c_x1 = torch.min(p_x1, g_x1)
    c_y1 = torch.min(p_y1, g_y1)
    c_x2 = torch.max(p_x2, g_x2)
    c_y2 = torch.max(p_y2, g_y2)

    c_w = (c_x2 - c_x1).clamp(min=0)
    c_h = (c_y2 - c_y1).clamp(min=0)
    c_area = c_w * c_h + eps

    # 6. Calculate GIoU & Loss
    giou = iou - ((c_area - union_area) / c_area)
    loss = 1.0 - giou

    return loss.mean()