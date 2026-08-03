from IoU import iou

def DIoU_loss(ground_truth, predicted):
    (g_x1, g_y1, g_x2, g_y2), (p_x1, p_y1, p_x2, p_y2) = ground_truth, predicted
    gt_center_x, gt_center_y = (g_x1 + g_x2) / 2, (g_y1 + g_y2) / 2
    p_center_x, p_center_y   = (p_x1 + p_x2) / 2, (p_y1 + p_y2) / 2
    d_between_centers = (gt_center_x - p_center_x) ** 2 + (gt_center_y - p_center_y) ** 2
    # co,pute smallest enclosing box
    c_x1 = min(p_x1, g_x1)
    c_y1 = min(p_y1, g_y1)
    c_x2 = max(p_x2, g_x2)
    c_y2 = max(p_y2, g_y2)

    diagonal_length = (c_x1 - c_x2) ** 2 + (c_y1 - c_y2) ** 2
    iou_value = iou(ground_truth, predicted)
    eps = 1e-7

    return 1 - iou_value + d_between_centers / (diagonal_length + eps)
