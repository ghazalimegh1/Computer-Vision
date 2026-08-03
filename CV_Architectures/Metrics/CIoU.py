from DIoU import DIoU_loss
from IoU import iou
import math

def CIoU_loss(ground_truth, predicted):
    (g_x1, g_y1, g_x2, g_y2), (p_x1, p_y1, p_x2, p_y2) = ground_truth, predicted
    w_gt, h_gt = abs(g_x1 - g_x2), abs(g_y1 - g_y2)
    w_p, h_p   = abs(p_x1 - p_x2), abs(p_y1 - p_y2)
    eps = 1e-7
    v = (4 / math.pi ** 2) * (math.atan(w_gt / (h_gt + eps)) - math.atan(w_p / (h_p + eps))) ** 2
    iou_value = iou(ground_truth, predicted)
    alpha = v / (1 - iou_value + v)
    return DIoU_loss(ground_truth, predicted) + alpha * v
