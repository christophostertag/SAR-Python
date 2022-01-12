#!/usr/bin/env python

import argparse
import os
from typing import Dict, List

from utils import BoundingBox, compute_AP, read_bb

parser = argparse.ArgumentParser(
    description="Evalute CV lab project score ws21/22.")
parser.add_argument('detections', type=str,
                    help='Detected bounding box dictionary.')
parser.add_argument('--set', type=str, default="validation",
                    help="Defines the sample set and must be " +
                         "in {validation, test}")
parser.add_argument('--dataset_root', type=str, default="./data",
                    help="Location of the dataset.")


def evaluate(detections: Dict[str, List[BoundingBox]],
             targets: Dict[str, List[BoundingBox]]) -> float:
    return compute_AP(detections, targets)


if __name__ == "__main__":
    args = parser.parse_args()

    assert args.set in ["validation", "test"]
    assert args.detections.endswith(".json")

    # load the detections
    detections = read_bb(args.detections)
    # load the targets
    targets = read_bb(os.path.join(args.dataset_root, args.set, "labels.json"))

    ap = evaluate(detections, targets)
    print(f"Average precision={ap:.5f} on {args.set} set.")
