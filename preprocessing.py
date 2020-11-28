import numpy as np
import os
from glob import glob
from typing import List, Dict, Tuple

# DEFINE FUNCTIONS

# bounding boxes are not always axis aligned
def aabb(polys: List[np.ndarray]) -> np.ndarray:
    bboxes = np.empty((len(polys), 4), dtype=np.float32)
    for i, poly in enumerate(polys):  # height, width
        # poly given as (x, y) points
        x = poly[:, 0]
        y = poly[:, 1]
        xmin, ymin = x.min(), y.min()
        xmax, ymax = x.max(), y.max()
        width = xmax - xmin
        height = ymax - ymin
        bboxes[i, :] = (xmin, ymin, width, height)
    return bboxes


def annotate_aabb(image, bboxes, color=(0, 255, 0)):
    image = image[..., None].repeat(3, axis=-1)  # rgb image
    for bbox in bboxes:
        image = cv2.rectangle(image, bbox, 
            color=color, thickness=2)
    return image  # annotation drawn on top

if __name__ == '__main__':
    # LOAD CALIBRATION PARAMETERS
    dir = os.path.join('calibration', 'parameters')

    err = np.load(os.path.join(dir,'err.npy'), allow_pickle=False)
    K = np.load(os.path.join(dir,'K.npy'), allow_pickle=False)
    dist_coeffs = np.load(os.path.join(dir,'dist_coeffs.npy'), allow_pickle=False)
    rvecs = np.load(os.path.join(dir,'rvecs.npy'), allow_pickle=False)
    tvecs = np.load(os.path.join(dir,'tvecs.npy'), allow_pickle=False)