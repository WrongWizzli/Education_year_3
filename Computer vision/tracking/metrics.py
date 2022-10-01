
def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4
    # Write code here
    import numpy as np
    max_y = max(bbox1[3], bbox2[3])
    max_x = max(bbox1[2], bbox2[2])
    a = np.zeros((max_x, max_y))
    b = np.zeros(a.shape)
    a[bbox1[0]:bbox1[2], bbox1[1]:bbox1[3]] += 1
    b[bbox2[0]:bbox2[2], bbox2[1]:bbox2[3]] += 1
    return np.logical_and(a, b).sum() / np.logical_or(a, b).sum()

def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs
    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        detections_obj = {}
        detections_hyp = {}
        # Step 1: Convert frame detections to dict with IDs as keys
        for elem in frame_obj:
            detections_obj[elem[0]] = elem[1:]
        for elem in frame_hyp:
            detections_hyp[elem[0]] = elem[1:]
        # Step 2: Iterate over all previous matches
        keys_hyp = list(detections_hyp.keys())
        keys_obj = list(detections_obj.keys())
        for key in keys_obj:
            if key in keys_hyp and iou_score(detections_obj[key], detections_hyp[key]) > threshold:
                match_count += 1
                matches[key] = key

                dist_sum += iou_score(detections_obj[key], detections_hyp[key])

                del detections_obj[key]
                del detections_hyp[key]
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        IoUs = []
        keys_obj = list(detections_obj.keys())
        keys_hyp = list(detections_hyp.keys())
        for key_obj in keys_obj:
            for key_hyp in keys_hyp:
                iou = iou_score(detections_obj[key_obj], detections_hyp[key_hyp])
                IoUs.append([iou, key_obj, key_hyp])

        # Save IDs with IOU > threshold
        # Step 4: Iterate over sorted pairwise IOU
        IoUs = sorted(IoUs, key=lambda x: x[0], reverse=True)
        mask_obj = {}
        mask_hyp = {}
        for IoU in IoUs:
            if IoU[0] > threshold and not IoU[1] in mask_obj.keys() and not IoU[2] in mask_hyp.keys():
                match_count += 1
                mask_obj[IoU[1]] = True
                mask_hyp[IoU[2]] = True
                matches[IoU[1]] = IoU[2]
                dist_sum += IoU[0]
                
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        # Step 5: Update matches with current matched IDs

    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count
    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0
    length = 0
    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame

    for frame_obj, frame_hyp in zip(obj, hyp):
        length += len(frame_obj)
        detections_obj = {}
        detections_hyp = {}
        # Step 1: Convert frame detections to dict with IDs as keys
        for elem in frame_obj:
            detections_obj[elem[0]] = elem[1:]
        for elem in frame_hyp:
            detections_hyp[elem[0]] = elem[1:]

        # Step 2: Iterate over all previous matches
        keys_hyp = list(detections_hyp.keys())
        keys_obj = list(detections_obj.keys())
        for key in keys_obj:
            if key in keys_hyp and iou_score(detections_obj[key], detections_hyp[key]) > threshold:
                match_count += 1
                matches[key] = key

                dist_sum += iou_score(detections_obj[key], detections_hyp[key])

                del detections_obj[key]
                del detections_hyp[key]
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        IoUs = []
        keys_obj = list(detections_obj.keys())
        keys_hyp = list(detections_hyp.keys())
        for key_obj in keys_obj:
            for key_hyp in keys_hyp:
                iou = iou_score(detections_obj[key_obj], detections_hyp[key_hyp])
                IoUs.append([iou, key_obj, key_hyp])

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        IoUs = sorted(IoUs, key=lambda x: x[0], reverse=True)
        mask_obj = {}
        mask_hyp = {}
        for IoU in IoUs:
            if IoU[0] > threshold and not IoU[1] in mask_obj.keys() and not IoU[2] in mask_hyp.keys():
                match_count += 1
                mask_obj[IoU[1]] = True
                mask_hyp[IoU[2]] = True
                matches[IoU[1]] = IoU[2]
                dist_sum += IoU[0]
                del detections_obj[IoU[1]]
                del detections_hyp[IoU[2]]

        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error

        # Step 6: Update matches with current matched IDs
        for key in matches.keys():
            if matches[key] != key:
                mismatch_error += 1
                matches[key] = key
        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        false_positive += len(detections_hyp)
        # All remaining objects are considered misses
        missed_count += len(detections_obj)

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count
    MOTA = 1 - (missed_count + false_positive + mismatch_error) / length

    return MOTP, MOTA
