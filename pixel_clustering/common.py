from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
from PIL import Image

from common import Paths, load_bounding_boxes, draw_bounding_boxes


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
