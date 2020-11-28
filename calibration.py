# IMPORTS
import argparse
import cv2
import os
import numpy as np
from typing import List, Dict, Tuple
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

# PARSE ARGS
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-v','--verbose', help='Show visual output of calibration', action='store_true')
args = vars(parser.parse_args())

verbose = args['verbose']

# FUNCTION DEFINITIONS
def show_images(images: np.ndarray, mask: np.ndarray = None, figsize=None):
    fig = plt.figure(figsize=figsize)
    if mask is not None:  # k,
        images = images[mask, ...]  # k x h x w
    images = torch.as_tensor(images).unsqueeze(1)  # n(k) x 1 x h x w
    grid = make_grid(images, nrow=int(np.sqrt(images.size(0))))  # 3 x h x w
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
    

def show_image(image: np.ndarray, cmap: str = None, figsize=None):
    fig = plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()

def search_for_corners(images, grid, invert=False) -> Tuple[np.array, np.ndarray]:
    found_corners = []
    mask = []
    for image in tqdm(images):
        if invert:
            image = -image + 255
        found, corners = cv2.findChessboardCorners(image, grid)
        mask.append(found)
        if found:  # refine corners on subpixel level
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
            found_corners.append(corners)
    return np.array(mask), np.array(found_corners)

def draw_corners(image, corners, grid, cmap=None):
    image = cv2.drawChessboardCorners(image[..., None].repeat(3, axis=-1), grid, corners, True)
    show_image(image, cmap=cmap)

### CALIBRATION

# READ IMAGES FOR CALIBRATION
print("LOADING IMAGES")
images = []
for image in glob('calibration/thermal/*.png'):
    images.append(cv2.imread(image, cv2.IMREAD_GRAYSCALE))  # h x w
images = np.array(images)  # n x h x w

if verbose:
    show_images(images[:16], figsize=(12, 12))

# checkerboard is of grid type (8, 7), as a convention
# the longer side is considered the width of the board
grid = (8, 7)
square_size = 50  # mm

# FIND CORNERS
print("SEARCHING FOR CHESS GRID CORNERS")
mask, found_corners = search_for_corners(images, grid, invert=True)
# Check if find corners failed complety
assert (mask == 1).any()
print(f'Found corners successfully on: {mask.sum()}/{len(mask)} images.')

if verbose:
    draw_corners(images[mask, ...][0], found_corners[0], grid)

image_points = found_corners  # 2d points in image plane
# 3d point in real world space, z coordinates will stay 0
world_points = np.zeros((*found_corners.shape[:2], 3), 
    dtype=found_corners.dtype)
# create xy world points
xy = square_size * np.mgrid[:grid[0],:grid[1]].T.reshape(-1, 2)
world_points[..., :2] = xy

# CALIBRATE
print("CALCULATING CAMERA CALIBRATION PARAMETERS")
err, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(world_points, 
    image_points, images.shape[-2:], None, None)

# SAVE PARAMS
print("SAVING CAMERA CALIBRATION PARAMETERS")
dir = os.path.join('calibration','parameters')

for par, par_name in zip([err, K, dist_coeffs, rvecs, tvecs],['err', 'K', 'dist_coeffs', 'rvecs', 'tvecs']):
    np.save(os.path.join(dir,par_name), par, allow_pickle=False)

print("FINISHED CALIBRATION")