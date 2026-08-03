# post-preprocessnig algorithm, before evaluating the model

from IoU import iou
import numpy as np


def nms(boxes, scores, iou_threshold=0.5):
    # sorting boxes
    sorted_boxes = sorted(zip(boxes, scores), key=lambda x: x[1], reverse=True)
    final_boxes = []
    while sorted_boxes:
        # getting the box with the highest score
        current_box, current_score = sorted_boxes.pop(0)
        # calcuating the IoU of the box with all other boxes
        final_boxes.append((current_box, current_score))
        iou_scores = [iou(current_box, box) for box, score in sorted_boxes]
        # remove boxes with IoU greater than the threshold
        sorted_boxes = [
            (box, score) for (box, score), iou_score in zip(sorted_boxes, iou_scores) if iou_score < iou_threshold
        ]

    return final_boxes
        





# to prevent models from missing objects that closely overlap each other, such as two people standing side-by-side in a crowd

def soft_nms(boxes, scores, iou_threshold=0.5, nms_threshold=0.001,method='linear', sigma=0.5): 
    if method not in ['linear', 'gaussian']:
        raise ValueError("Method must be either 'linear' or 'gaussian'")

    # sorting boxes
    sorted_boxes = sorted(zip(boxes, scores), key=lambda x: x[1], reverse=True)
    final_boxes = []
    while sorted_boxes:
        current_box, current_score = sorted_boxes.pop(0)
        if current_score < nms_threshold:
            break
        final_boxes.append((current_box, current_score))
        iou_scores = [iou(current_box, box) for box, score in sorted_boxes]

        updated_boxes = []
        for i, (box, score) in enumerate(sorted_boxes):
            if iou_scores[i] > iou_threshold:
                if method == 'linear':
                    score = (box, score * (1 - iou_scores[i]))
                elif method == 'gaussian':
                    exponent = -(iou_scores[i] ** 2) / (sigma)
                    score = (box, np.exp(exponent) * score)
                if score >= nms_threshold:
                    updated_boxes.append((box, score))
        
        sorted_boxes = sorted(updated_boxes, key=lambda x: x[1], reverse=True)

        

    return final_boxes
                


# class-wise nms run nms independently for each class
def class_wise_nms(boxes, scores, iou_threshold=0.5):
    unique_classes = {}
    for box, score in zip(boxes, scores):
        box_class = box.class_id
        if box_class not in unique_classes:
            unique_classes[box_class] = []
        unique_classes[box_class].append((box, score))

    results = []
    for cls, detections in unique_classes.items():
        boxes_ = [b for b, s in detections]
        scores_ = [s for b, s in detections]
        kept = nms(boxes_, scores_, iou_threshold)
        results.extend(kept)

    return results
