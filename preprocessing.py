# IMPORTS
# external
import numpy as np
import os
import cv2
from glob import glob
import json
from typing import List, Dict, Tuple
# internal
from calibration import show_image, show_images

# DEFINE FUNCTIONS

def imread(files: List[str], undistort=False):
    images = []
    for path in files:
        image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        image = cv2.normalize(image, dst=None, alpha=0, beta=2**16 - 1, 
            norm_type=cv2.NORM_MINMAX)
        image = (image >> 8).astype(np.uint8)  # 8 bit image
        if undistort:  # undistort camera images
            h, w = image.shape
            # returns the new camera intrinsic matrix based 
            # on the free scaling parameter
            refined_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, 
                (w, h), 1, (w, h))
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]
            image = cv2.undistort(image, K, dist_coeffs, None, refined_K)
        images.append(image)
    images = np.array(images)  # n x h x w
    return images

def integrate(images: np.ndarray, win: List[int], K: np.ndarray, M: np.ndarray, z: float, verbose=False):
    """ win ... window of integration """
    h, w = images.shape[1:]  # gray scale images
    integral = np.zeros((h, w), np.float64)
    _images = images[win]
    _M = M[win]
    count = len(win)

    # center index
    idc = len(_images) // 2

    # invese of the intrinsic mapping
    K_inv = np.linalg.inv(K)

    # M = ( A b ) or M = K ( R t )
    Mc = _M[idc]  # 3 x 4

    # the given M matrices seem to consist solely of ( R t ) part
    Rc = Mc[:, :3]  # 3 x 3
    tc = Mc[:, 3:]  # 3 x 1

    for i in range(len(_images)):
        if i != idc:
            Mr = _M[i]  # 3 x 4
            Rr = Mr[:, :3]  # 3 x 3
            tr = Mr[:, 3:]  # 3 x 1

            # relative translation and rotation
            R_ = Rc @ Rr.T  # 3 x 3
            t_ = tc - R_ @ tr  # 3 x 1

            B = K @ R_ @ K_inv
            B[:, 2:] += K @ t_ / z
            warped = cv2.warpPerspective(_images[i], B, (w, h))
            if verbose:
                show_image(warped, cmap='gray')
            integral += warped
        else:
            if verbose:
                show_image(_images[i], cmap='gray')
            integral += _images[i]

    integral /= count
    integral = cv2.normalize(integral, None, 
        0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return integral  # 8bit gray scale integral image 

def find_file_index(files: List[str], name: str):
    """ return -1 if name not part of a file name"""
    for i, file in enumerate(files):
        if name in file:
            return i
    else:
        return -1

def annotate(image, polys, color=(0, 255, 0)):
    image = image[..., None].repeat(3, axis=-1)  # rgb image
    polys = np.array(polys, dtype=np.int32)
    image = cv2.polylines(image, polys, isClosed=True, 
        color=color, thickness=2)
    return image  # annotation drawn on top

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

    #err = np.load(os.path.join(dir,'err.npy'), allow_pickle=False)
    K = np.load(os.path.join(dir,'K.npy'), allow_pickle=False)
    #dist_coeffs = np.load(os.path.join(dir,'dist_coeffs.npy'), allow_pickle=False)
    #rvecs = np.load(os.path.join(dir,'rvecs.npy'), allow_pickle=False)
    #tvecs = np.load(os.path.join(dir,'tvecs.npy'), allow_pickle=False)