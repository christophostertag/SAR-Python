import json
from pathlib import Path
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
from PIL import Image


class Paths:
    data = Path(__file__).parent / 'data'

    data_train = data / 'train'
    data_test = data / 'test'
    validation_labels = data / 'validation/labels.json'

    output = Path.home() / 'data_WiSAR/output1'


def load_mask(mask_path=Paths.data / 'mask.png'):
    mask = np.array(Image.open(mask_path))[:, :, None]
    mask = mask / mask.max()
    mask = mask.astype(np.uint8)
    return mask


def load_homographies(
        dir: Path,
):
    with open(dir / 'homographies.json', 'r') as h:
        homographies = json.load(h)
    return homographies


def load_bounding_boxes(path=Paths.validation_labels) -> Dict[str, np.ndarray]:
    with open(path, 'r') as h:
        j = json.load(h)
    return {k: np.array(j[k]) for k in sorted(j.keys())}


def draw_bounding_boxes(
        image: np.ndarray,
        boxes: np.ndarray,
        box_color=(0, 255, 255),
        line_thickness=2,
        x_offset=0,
        y_offset=0,
):
    t = line_thickness
    for y0, x0, ly, lx in boxes:
        x0 += x_offset
        y0 += y_offset
        image[x0:x0 + lx, y0:y0 + t] = box_color
        image[x0:x0 + lx, y0 + ly - t:y0 + ly] = box_color
        image[x0:x0 + t, y0:y0 + ly] = box_color
        image[x0 + lx:x0 + lx + t, y0:y0 + ly] = box_color


def get_image_sets(
        dir=Paths.data,
        filter='train',
        mask=True,
        warp=True,
        timesteps=range(-3, 4),
        views=range(10),
        cropx=192,
        size_filter: Optional[Tuple[int]] = (1024, 1024, 3),
        sets=1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_views = 10
    timesteps = np.array(timesteps)
    views = np.array(views)

    all_bounding_boxes = load_bounding_boxes()

    image_dirs = [d for d in dir.glob('*/*') if d.is_dir() and filter in d.as_posix()]
    mask_ = load_mask()
    mask_ = np.concatenate([mask_] * 3, axis=2)

    for i, image_dir in enumerate(image_dirs):
        if i >= sets:
            break
        homographies = load_homographies(image_dir)
        bounding_boxes = all_bounding_boxes.get(image_dir.parts[-1], np.array([])).copy()

        # adjust boxes when crop was set!
        if len(bounding_boxes):
            bounding_boxes[:, 1] -= cropx

        image_paths = image_dir.glob('*.png')
        image_paths = [p.relative_to(dir) for p in image_paths]
        image_paths = np.array(image_paths).reshape(-1, n_views)
        image_paths = image_paths[timesteps, views[:, None]]

        images = []

        for file in image_paths.reshape(-1):
            image = cv2.imread(str(dir / file))
            # with Image.open(file) as im:
            #     image2 = np.array(im)
            if size_filter is None or image.shape == size_filter:
                w, h, _ = image.shape
                if warp:
                    image_name = file.name.replace('.png', '')
                    homography = np.array(homographies[image_name])
                    image = cv2.warpPerspective(image, homography, (w, h))
                    # image = cv2.bitwise_and(image, mask)
                if mask:
                    image *= mask_
                if cropx > 0:
                    image = image[cropx:-cropx]

                images.append(image)

        images = np.array(images)
        images = np.array(images).reshape(len(timesteps), len(views), *images.shape[1:])
        yield images, image_paths, bounding_boxes


def demo_load_image_sets():
    image_sets = get_image_sets(sets=2, filter='valid')
    i = 0
    for images, paths, boxes in image_sets:
        im = images.mean(1).astype(np.uint8)[0]
        draw_bounding_boxes(im, boxes)
        print(f'Shape of image set {i}: {images.shape}, bounding_boxes={len(boxes) > 0}')

        cv2.imshow("Window", im)
        cv2.waitKey()

        i += 1


if __name__ == '__main__':
    demo_load_image_sets()
