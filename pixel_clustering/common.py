import json
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
from PIL import Image

from common import Paths


def load_image_batches(
        path=Paths.data,
        batch_size=32,
        filter='train',
        size: Optional[Tuple[int]] = (1024, 1024, 3),
):
    all_paths = path.glob('**/*.png')
    all_paths = [Path(p) for p in all_paths if
                 filter in p.as_posix() and not p.as_posix().endswith('mask.png')]

    done = False

    while not done:
        images = []
        path_list = []
        done = True
        for p in all_paths:
            if len(images) >= batch_size:
                done = False
                break

            with Image.open(p) as im:
                image = np.array(im)
            if size is None or image.shape == size:
                images.append(image)
                path_list.append(p)
        yield np.array(images), path_list


def load_mask(mask_path=Paths.data / 'mask.png'):
    mask = np.array(Image.open(mask_path))[None, :, :, None]
    mask = mask / mask.max()
    mask = mask.astype(np.uint8)
    return mask


def load_bounding_boxes(path=Paths.validation_labels) -> Dict[str, np.ndarray]:
    with open(path, 'r') as h:
        j = json.load(h)
    return {k: np.array(j[k]) for k in sorted(j.keys())}


def crop190(images: np.ndarray) -> np.ndarray:
    return images[:, 190:-190]


def get_centered_rectangle_slice(
        tensor: np.ndarray,
        dimensions: np.ndarray,
):
    """Returns a centered slice of the tensor with given dimensions."""
    outer_lengths = tensor.shape[:len(dimensions)]
    inner_lengths = dimensions

    offsets = (np.array(outer_lengths) - np.array(inner_lengths)) // 2

    slices = []
    for offset, length in zip(offsets, inner_lengths):
        slices.append(slice(offset, offset + length))

    return tensor[tuple(slices)]


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


def select_bounding_boxes(
        image_path: Optional[Path],
        bounding_boxes: Dict[str, np.ndarray],
) -> np.ndarray:
    if image_path.parts[-1].endswith('B01.png'):
        key = image_path.parts[-2]
        if key in bounding_boxes:
            return bounding_boxes[key]
    return np.empty(shape=(0, 4))


def test_bounding_boxes():
    bounding_boxes = load_bounding_boxes()
    image_batches = load_image_batches(path=Paths.data, batch_size=770, filter='validation')
    images, paths = next(image_batches)
    rel_paths = [p.relative_to(Paths.data) for p in paths]
    images = crop190(images)

    for image, rel_path in zip(images, rel_paths):
        boxes = select_bounding_boxes(rel_path, bounding_boxes)
        draw_bounding_boxes(image, boxes, x_offset=-190)
        p = Paths.output / 'boxes' / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        # Image.fromarray(image).save(p)


if __name__ == '__main__':
    test_bounding_boxes()
