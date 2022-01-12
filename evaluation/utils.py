#!/usr/bin/env python

import json
import numpy as np

from typing import Dict, NewType, List, Tuple, Union
import pathlib

Path = Union[pathlib.Path, str]
BoundingBox = NewType("BoundingBox", Tuple[int, int, int, int])
YoloBox = NewType("YoloBox", Tuple[float, float, float, float])
Shape = NewType("Shape", Tuple[int, int])


def compute_AP(detections: Dict[str, List[BoundingBox]],
               targets: Dict[str, List[BoundingBox]]) -> float:
    """ Compute the average precision.

    Params:
        detections: list of detected bounding boxes within each sample
        targets: list of ground truth bounding boxes within each sample
    """
    # define the IoU threshold sequence
    thresholds = np.arange(0.1, 1.0, 0.1)

    precision = np.zeros_like(thresholds)
    recall = np.zeros_like(thresholds)

    iou_scores = [compute_IoU(detections[k], targets[k])
                  for k in targets.keys()]

    for i, iou_th in enumerate(thresholds):
        true_positives = sum(
            [np.sum(np.any(iou > iou_th, 1)) for iou in iou_scores])
        false_positives = sum(
            [np.sum(~np.any(iou > iou_th, 1)) for iou in iou_scores])
        false_negatives = sum(
            [np.sum(~np.any(iou > iou_th, 0)) for iou in iou_scores])

        if true_positives + false_positives:
            precision[i] = true_positives/(true_positives+false_positives)
        else:
            precision[i] = 0
        recall[i] = true_positives/(true_positives+false_negatives)

    # compute average precision
    recall = np.append(recall, 0)
    ap = np.sum((recall[:-1] - recall[1:]) * precision)
    return ap


def compute_IoU(detections: List[BoundingBox],
                targets: List[BoundingBox]) -> np.array:
    """ Compute the intersection of union (IoU) score.

    Params:
        detections: detected bounding boxes
        targets: ground truth bounding boxes

    Return:
        Array of IoU score between each pair of detected and target bounding
        box, where the detections are along the rows and the targets along
        the columns.
    """
    iou = np.empty((len(detections), len(targets)))

    for i, d in enumerate(detections):
        dx, dy, dw, dh = d
        for j, t in enumerate(targets):
            tx, ty, tw, th = t
            x = max(dx, tx)
            y = max(dy, ty)
            xx = min(dx + dw, tx + tw)
            yy = min(dy + dh, ty + th)
            intersection_area = max(0, xx-x) * max(0, yy-y)
            iou[i, j] = intersection_area / (dw*dh + tw*th - intersection_area)
    return iou


def read_bb(file: Path) -> Dict[str, List[BoundingBox]]:
    """ Read bounding boxes from json file.
    """
    with open(file) as f:
        js = json.load(f)
    return js


def write_bb(file: Path, bbs: Dict[str, List[BoundingBox]]) -> None:
    """ Write bounding boxes to json file.
    """
    with open(file, "w") as f:
        json.dump(bbs, f)
