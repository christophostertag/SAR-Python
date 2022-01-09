import cv2
import numpy as np
from scipy.signal import convolve2d

from common import draw_bounding_boxes, Paths, get_image_sets, imshow, find_boxes


def get_temporal_diff_heatmaps(
        images: np.ndarray,
        boxes: np.ndarray,
        blur_color_diff=lambda x: cv2.GaussianBlur(x, (19, 19), 0),

):
    fl_images = images.astype(np.float64)
    fl_images[fl_images.sum(-1) == 0] = np.nan

    heatmaps = []

    for step in range(0, 7):
        other_steps = list(range(0, 7))
        other_steps.remove(step)

        color_diff = fl_images[step] - np.nanmean(fl_images[other_steps], axis=0)
        color_diff_filtered = color_diff.copy()
        for i in range(len(color_diff)):
            color_diff_filtered[i] = blur_color_diff(color_diff_filtered[i])

        # for cdiff in color_diff_filtered[6:]:
        #     cdiff = cdiff.copy()
        #     draw_bounding_boxes(cdiff, boxes)
        #     imshow(cdiff * 2 + 128)

        # heatmaps_row = np.linalg.norm(color_diff_filtered, axis=-1)

        average_color_diff = np.nanmean(color_diff, axis=(1, 2), keepdims=True)
        average_color_diff = 0
        heatmaps_row = np.linalg.norm(color_diff_filtered - average_color_diff, ord=2, axis=-1)
        heatmaps_row = np.nan_to_num(heatmaps_row, nan=0)

        heatmaps.append(heatmaps_row)

    heatmaps = np.array(heatmaps)

    return heatmaps


def get_detection_map(
        heatmaps: np.ndarray,
        kernel_size=19,
        thresh_quantile=0.95,
        thresh_factor=2,
        min_detections=7,
) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2

    detection_maps = []
    for heatmap in heatmaps.reshape(-1, *heatmaps.shape[2:]):
        conv_heatmap = convolve2d(heatmap, kernel, mode='same')
        thresh = thresh_factor * np.quantile(conv_heatmap, q=thresh_quantile)
        detection = conv_heatmap > thresh
        detection_maps.append(detection)

    detection_maps = np.reshape(detection_maps, heatmaps.shape)
    detection_map = detection_maps.sum(axis=(0, 1)) >= min_detections

    return detection_map


def main(
        show=True,
):
    image_sets = get_image_sets(sets=100, filter='valid')
    for images, paths, boxes in image_sets:
        heatmaps = get_temporal_diff_heatmaps(images, boxes)
        detection_map = get_detection_map(heatmaps)
        im = images[3, 4].copy()
        im[np.where(detection_map > 0)] = [0, 0, 255]
        ourboxes = find_boxes(detection_map)
        draw_bounding_boxes(im, boxes)
        draw_bounding_boxes(im, ourboxes, box_color=(255, 255, 0))
        p = Paths.output / paths[3, 4]
        p.parent.mkdir(parents=True, exist_ok=True)
        # cv2.imwrite(str(p), im)
        print(f'Image exported: {p}')
        if show:
            imshow(im)


if __name__ == '__main__':
    main()
