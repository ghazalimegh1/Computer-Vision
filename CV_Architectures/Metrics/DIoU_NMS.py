from DIoU import DIoU_loss

def DIoU_nms(boxes, scores, diounms_threshold):
    # sorting boxes
    sorted_boxes = sorted(zip(boxes, scores), key=lambda x: x[1], reverse=True)
    final_boxes = []
    while sorted_boxes:
        # getting the box with the highest score
        current_box, current_score = sorted_boxes.pop(0)
        final_boxes.append((current_box, current_score))
        diou_scores = [1 - DIoU_loss(current_box, box) for box, score in sorted_boxes]
        # Remove boxes whose DIoU similarity exceeds the threshold        
        sorted_boxes = [
            (box, score) for (box, score), diou_score in zip(sorted_boxes, diou_scores) if diou_score < diounms_threshold
        ]

    return final_boxes