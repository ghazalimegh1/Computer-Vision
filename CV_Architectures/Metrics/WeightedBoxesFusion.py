from Metrics.IoU import iou

def WBF(boxes, confidence_scores, iou_threshold=0.55):
    # B list
    N = len(boxes)
    B = list(zip(boxes, confidence_scores))
    # sorting descending order by confidence score
    B.sort(key=lambda x:x[1], reverse=True)
    # declare empty lsits L (clusters) and F (fused box, score)
    L = []
    F = []
    # recursive calling
    while B:
        # take element from B, find the matching
        box, confidence_score = B.pop(0) 
        matching = False
        current_iou = None
        for i, (fused_box, score) in enumerate(F):
            # if found, add it to the corresponding cluster, and update the fusion box
            iou_value = iou(box, fused_box)
            if iou_value > iou_threshold:
                matching = True
                if current_iou is None or  iou_value > current_iou[0]:
                    current_iou = (iou_value, i) 


        if matching:
            index = current_iou[1]
            L[index].append((box, confidence_score))
            F[index] = calculate_fused_box(L[index])
                


        # if not found --> new cluster
        if not matching:
            L.append([(box, confidence_score)])
            F.append([box, confidence_score])

    # rescale confidence score
    for l, f in zip(L, F):
        f[1] *= (min(N, len(l)) / N) 

    return F




    
    

def calculate_fused_box(cluster):
    
    sum_ci = sum(score for box, score in cluster)
    T = len(cluster)
    C = sum_ci / T
    x_min = y_min = x_max = y_max = 0.0
    for i, (box, score) in enumerate(cluster):
        x_min += box[0] * score
        y_min += box[1] * score
        x_max += box[2] * score
        y_max += box[3] * score

    x_min /= sum_ci
    y_min /= sum_ci
    x_max /= sum_ci
    y_max /= sum_ci

    fused_box = (x_min, y_min, x_max, y_max)

    return [fused_box, C]