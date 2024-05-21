# if has_intersect:
# 	if bb.id == bb_1.id:
# 		if center_dist < max(0.5*sqrt(w1^2+h1^2), 0.5*sqrt(w1^2+h1^2)):
# 			new_bb = merge(bb, bb_1)
# 	else:
# 		continue
#
# def merge(bb, bb_1):
# 	new_bb.id = bb.id
# 	new_bb.center = 0.5*(bb.x+bb_1.x, bb.y+bb_1.y)	# taking consideration of multiple detections
# 	if bb.conf > bb_1.conf:
# 		new_bb.w = bb.w
# 		new_bb.h = bb.h
# 	else:
# 		new_bb.w = bb_1.w
# 		new_bb.h = bb_1.h
# 	return new_bb


import copy
import math
import numpy as np


def xywh_to_corners(bbox):
    x_min = bbox[1][0] - 0.5 * bbox[1][2]
    y_min = bbox[1][1] - 0.5 * bbox[1][3]
    x_max = bbox[1][0] + 0.5 * bbox[1][2]
    y_max = bbox[1][1] + 0.5 * bbox[1][3]
    return x_min, y_min, x_max, y_max


def is_in_bbox(point, bbox):
    """
    Arguments:
        point {list} -- list of float values (x,y)
        bbox {list} -- bounding box of float_values [id, [x_c, y_c, w, h], conf]
    Returns:
        {boolean} -- true if the point is inside the bbox
    """
    x_min, y_min, x_max, y_max = xywh_to_corners(bbox)
    return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max


def intersect(bbox, bbox_):
    """
    Arguments:
        bbox {list} -- bounding box of float_values [id, [x_c, y_c, w, h], conf]
        bbox_ {list} -- bounding box of float_values [id, [x_c, y_c, w, h], conf]
    Returns:
        {boolean} -- true if the bboxes intersect
    """
    # Check if one of the corner of bbox inside bbox_
    corner_1 = [bbox[1][0] - 0.5 * bbox[1][2], bbox[1][1] - 0.5 * bbox[1][3]]
    corner_2 = [bbox[1][0] - 0.5 * bbox[1][2], bbox[1][1] + 0.5 * bbox[1][3]]
    corner_3 = [bbox[1][0] + 0.5 * bbox[1][2], bbox[1][1] - 0.5 * bbox[1][3]]
    corner_4 = [bbox[1][0] + 0.5 * bbox[1][2], bbox[1][1] + 0.5 * bbox[1][3]]
    corners = [corner_1, corner_2, corner_3, corner_4]
    for i in range(len(corners)):
        if is_in_bbox(corners[i], bbox_):
            return True
    return False


def dynamic_merge(bbox, bbox_):
    new_bbox = copy.deepcopy(bbox)
    new_bbox[0] = bbox[0]

    # Strategy 1
    # weight = bbox[2] / (bbox[2] + bbox_[2])
    # weight_ = bbox_[2] / (bbox[2] + bbox_[2])
    # new_bbox[1][0] = weight * bbox[1][0] + weight_ * bbox_[1][0]
    # new_bbox[1][1] = weight * bbox[1][1] + weight_ * bbox_[1][1]  # taking consideration of multiple detections
    # new_bbox[1][2] = bbox[2] * bbox[1][2] + bbox_[2] * bbox_[1][2]  # merge to generate bigger bounding boxes
    # new_bbox[1][3] = bbox[2] * bbox[1][3] + bbox_[2] * bbox_[1][3]
    # new_bbox[2] = weight * bbox[2] + weight_ * bbox_[2]

    # Strategy 2
    x_min, y_min, x_max, y_max = xywh_to_corners(bbox)
    x_min_, y_min_, x_max_, y_max_ = xywh_to_corners(bbox_)
    x_min_new = min(x_min, x_min_)
    y_min_new = min(y_min, y_min_)
    x_max_new = max(x_max, x_max_)
    y_max_new = max(y_max, y_max_)
    new_bbox[1][0] = 0.5 * (x_min_new + x_max_new)
    new_bbox[1][1] = 0.5 * (y_min_new + y_max_new)
    new_bbox[1][2] = x_max_new - x_min_new
    new_bbox[1][3] = y_max_new - y_min_new
    new_bbox[2] = bbox[2] * bbox[2] + bbox_[2] * bbox_[2]   # use confidence as a weight (or can just average)

    return new_bbox


def merge_bboxes(bboxes, delta_x=0.05, delta_y=0.05):
    """
    Arguments:
        bboxes {list} -- list of bounding boxes with each bounding box is a list [id, [x_c, y_c, w, h], conf]
        delta_x {float} -- margin taken in width to merge
        delta_y {float} -- margin taken in height to merge
    Returns:
        {list} -- list of bounding boxes merged
    """

    # Sort bboxes by x_c
    bboxes = sorted(bboxes, key=lambda x: x[1][0])
    # print(bboxes)

    tmp_bbox = None
    while True:
        nb_merge = 0
        used = []
        new_bboxes = []
        # Loop over bboxes
        for i, b in enumerate(bboxes):
            for j, b_ in enumerate(bboxes):
                # If the bbox has already been used just continue
                if i in used or j <= i:
                    continue
                if b[0] == b_[0]:  # merge only happens for bounding boxes with the same ID
                    # Compute the bboxes with a margin
                    b_margin = copy.deepcopy(b)
                    # print(b)
                    b_margin[1][2] = b_margin[1][2] * (1 + 2 * delta_x)
                    b_margin[1][3] = b_margin[1][3] * (1 + 2 * delta_y)
                    # print(b_margin)
                    # print(b)
                    b_margin_ = copy.deepcopy(b_)
                    b_margin_[1][2] = b_margin_[1][2] * (1 + 2 * delta_x)
                    b_margin_[1][3] = b_margin_[1][3] * (1 + 2 * delta_y)
                    # Merge bboxes if bboxes with margin have an intersection and center distance meets certain requirement
                    # Check if one of the corner is in the other bbox
                    # We must verify the other side away in case one bounding box is inside the other
                    if intersect(b_margin, b_margin_) or intersect(b_margin_, b_margin):
                        center_dist = math.dist([b[1][0], b[1][1]], [b_[1][0], b_[1][1]])
                        # print("center_dist = " + str(center_dist))
                        # print("threshold = " + str(max(np.sqrt((0.5 * b[1][2]) * (0.5 * b[1][2]) + (0.5 * b[1][3]) * (0.5 * b[1][3])),
                        #                                np.sqrt((0.5 * b_[1][2]) * (0.5 * b_[1][2]) + (0.5 * b_[1][3]) * (0.5 * b_[1][3])))))
                        if center_dist < max(np.sqrt((0.5 * b[1][2]) * (0.5 * b[1][2]) + (0.5 * b[1][3]) * (0.5 * b[1][3])),
                                             np.sqrt((0.5 * b_[1][2]) * (0.5 * b_[1][2]) + (0.5 * b_[1][3]) * (0.5 * b_[1][3]))):
                            tmp_bbox = dynamic_merge(b, b_)
                            used.append(j)
                            # print(b_margin, b_margin_, 'done')
                            nb_merge += 1
                    if tmp_bbox:
                        b = tmp_bbox
            if tmp_bbox:
                new_bboxes.append(tmp_bbox)
            elif i not in used:
                new_bboxes.append(b)
            used.append(i)
            tmp_bbox = None
        # If no merge were done, that means all bboxes were already merged
        if nb_merge == 0:
            break
        bboxes = copy.deepcopy(new_bboxes)

    return new_bboxes
