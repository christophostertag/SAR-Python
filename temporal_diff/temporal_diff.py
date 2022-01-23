import json
from typing import Tuple

import cv2
import numpy as np
from scipy.signal import convolve2d

from common import draw_bounding_boxes, Paths, get_image_sets, imshow, find_boxes
from evaluation.utils import compute_AP


def get_temporal_diff_heatmaps(
        images: np.ndarray,
        boxes: np.ndarray,
        blur_color_diff=lambda x: cv2.GaussianBlur(x, (19, 19), 0),
):
    fl_images = images.astype(np.float64)
    # we set regions outside the mask to nan and use np.nanmean, np.nanquantile, etc. later
    fl_images[fl_images.sum(-1) == 0] = np.nan

    heatmaps = []

    for step in range(0, 7):
        other_steps = list(range(0, 7))
        other_steps.remove(step)

        color_diff = fl_images[step] - np.nanmean(fl_images[other_steps], axis=0)
        color_diff_filtered = color_diff.copy()
        for i in range(len(color_diff)):
            color_diff_filtered[i] = blur_color_diff(color_diff_filtered[i] / 255)

        # cdiff = color_diff_filtered[7] * 2 * 255 + 128
        # draw_bounding_boxes(cdiff, boxes)
        # imshow(cdiff)

        # color_diff_correction = np.nanmean(color_diff, axis=(1, 2), keepdims=True)
        color_diff_correction = 0
        heatmaps_row = np.linalg.norm(color_diff_filtered - color_diff_correction, ord=2, axis=-1)
        heatmaps_row = np.nan_to_num(heatmaps_row, nan=0)
        heatmaps.append(heatmaps_row)

        # luminosity = (np.maximum(np.maximum(color_diff_filtered[:, :, :, 0], color_diff_filtered[:, :, :, 1]), color_diff_filtered[:, :, :, 2]) + np.minimum(np.minimum(color_diff_filtered[:, :, :, 0], color_diff_filtered[:, :, :, 1]), color_diff_filtered[:, :, :, 2])) * 0.5
        # saturation = (np.maximum(np.maximum(color_diff_filtered[:, :, :, 0], color_diff_filtered[:, :, :, 1]), color_diff_filtered[:, :, :, 2]) - np.minimum(np.minimum(color_diff_filtered[:, :, :, 0], color_diff_filtered[:, :, :, 1]), color_diff_filtered[:, :, :, 2])) / (1 - np.abs(luminosity * 2 - 1))
        # heatmaps.append(np.nan_to_num(saturation))
        #
        # weight_map = np.empty((1024, 1024, 3))
        # weight_map[:, :, 0] = saturation[7] * 255
        # weight_map[:, :, 1] = weight_map[:, :, 0]
        # weight_map[:, :, 2] = weight_map[:, :, 0]
        # draw_bounding_boxes(weight_map, boxes, box_color=(0, 170, 255))
        # imshow(weight_map)

        # hm = heatmaps_row[7]
        # hm = np.moveaxis((hm, hm, hm), 0, 2) * 2
        # draw_bounding_boxes(hm, boxes)
        # imshow(hm * 255)

    heatmaps = np.array(heatmaps)
    return heatmaps


def find_clusters(detection_map):
    detection_map = detection_map.copy()
    cluster_positions = []
    cluster_intensities = []
    for x in range(detection_map.shape[0]):
        if detection_map[x].any() != 0:
            for y in range(detection_map.shape[1]):
                todo = []
                positions = []
                if detection_map[x, y] != 0:
                    todo.append((x, y))
                    positions.append((x, y))

                    value = 0
                    while len(todo) != 0:
                        x, y = todo.pop()
                        if value < detection_map[x, y]:
                            value = detection_map[x, y]
                            todo = []
                            positions = [(x, y)]

                        if (x + 1, y) not in positions and detection_map[x + 1, y] != 0 and detection_map[x + 1, y] >= value:
                            todo.append((x + 1, y))
                            positions.append((x + 1, y))
                        if (x - 1, y) not in positions and detection_map[x - 1, y] != 0 and detection_map[x - 1, y] >= value:
                            todo.append((x - 1, y))
                            positions.append((x - 1, y))
                        if (x, y + 1) not in positions and detection_map[x, y + 1] != 0 and detection_map[x, y + 1] >= value:
                            todo.append((x, y + 1))
                            positions.append((x, y + 1))
                        if (x, y - 1) not in positions and detection_map[x, y - 1] != 0 and detection_map[x, y - 1] >= value:
                            todo.append((x, y - 1))
                            positions.append((x, y - 1))

                    intensity = value
                    position = positions[0]

                    cluster_positions.append(position)
                    cluster_intensities.append(intensity)

                    todo = [position]
                    positions = [position]

                    while len(todo) != 0:
                        x, y = todo.pop()
                        value = detection_map[x, y]
                        detection_map[x, y] = 0

                        if (x + 1, y) not in positions and detection_map[x + 1, y] != 0 and detection_map[x + 1, y] <= value:
                            todo.append((x + 1, y))
                            positions.append((x + 1, y))
                        if (x - 1, y) not in positions and detection_map[x - 1, y] != 0 and detection_map[x - 1, y] <= value:
                            todo.append((x - 1, y))
                            positions.append((x - 1, y))
                        if (x, y + 1) not in positions and detection_map[x, y + 1] != 0 and detection_map[x, y + 1] <= value:
                            todo.append((x, y + 1))
                            positions.append((x, y + 1))
                        if (x, y - 1) not in positions and detection_map[x, y - 1] != 0 and detection_map[x, y - 1] <= value:
                            todo.append((x, y - 1))
                            positions.append((x, y - 1))

    return cluster_positions, cluster_intensities


def keep_top_n(detection_map, cluster_positions, cluster_intensities, min_detections, MIN_PIXELS=100):
    label = np.zeros_like(detection_map, dtype=bool)
    sorted_indices = np.argsort(cluster_intensities)
    for i in reversed(sorted_indices):
        todo = [cluster_positions[i]]
        positions = [cluster_positions[i]]

        while len(todo) != 0:
            x, y = todo.pop()
            value = detection_map[x, y]

            if label[x, y]:
                positions =[]
                break

            if (x + 1, y) not in positions and detection_map[x + 1, y] >= min_detections:
                todo.append((x + 1, y))
                positions.append((x + 1, y))
            if (x - 1, y) not in positions and detection_map[x - 1, y] >= min_detections:
                todo.append((x - 1, y))
                positions.append((x - 1, y))
            if (x, y + 1) not in positions and detection_map[x, y + 1] >= min_detections:
                todo.append((x, y + 1))
                positions.append((x, y + 1))
            if (x, y - 1) not in positions and detection_map[x, y - 1] >= min_detections:
                todo.append((x, y - 1))
                positions.append((x, y - 1))

        if len(positions) > MIN_PIXELS:

            positions = np.swapaxes(np.array(positions), 0, 1)

            weights = detection_map[positions[0], positions[1]]
            average = (np.average(positions[0], weights=weights), np.average(positions[1], weights=weights))
            distance = np.linalg.norm((positions[0] - average[0], positions[1] - average[1]), axis=0)
            order = weights / (1 + np.sqrt(distance) * 0.2)

            for distance_index in np.argsort(distance)[MIN_PIXELS:]:
                if order[distance_index] < cluster_intensities[i] / 4:
                    positions[:, distance_index] = -1

            positions = positions[:, positions[0] >= 0]

            label[positions[0], positions[1]] = True

    return label


def get_detection_map(
        heatmaps: np.ndarray,
        images:np.ndarray,
        kernel_size=19,
        thresh_quantile=0.95,
        thresh_factor=2,
        min_detections=7,
        boxes=[],
        debug=False,
) -> Tuple[np.ndarray, np.ndarray]:
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2

    detection_maps = []
    for heatmap in heatmaps.reshape(-1, *heatmaps.shape[2:]):
        conv_heatmap = convolve2d(heatmap, kernel, mode='same')
        thresh = thresh_factor * np.nanquantile(conv_heatmap, thresh_quantile)
        detection = conv_heatmap > thresh
        detection_maps.append(detection)

    detection_maps = np.reshape(detection_maps, heatmaps.shape)

    # detection_map = detection_maps.sum(axis=(0, 1)) >= min_detections
    detection_map = detection_maps.sum(axis=(0, 1))

    if debug:
        detection_map_bb = detection_map.copy()
        draw_bounding_boxes(np.moveaxis([detection_map_bb] * 3, 0, 2), boxes, box_color=(0, 170, 255))
        imshow(detection_map_bb*255/20)

    cluster_positions, cluster_intensities = find_clusters(detection_map)

    for i in reversed(range(len(cluster_intensities))):
        if cluster_intensities[i] < min_detections:
            cluster_positions.pop(i)
            cluster_intensities.pop(i)

    label = keep_top_n(detection_map, cluster_positions, cluster_intensities, min_detections, 100)

    ourboxes = find_boxes(label)

    return label, ourboxes


def blor(image:np.ndarray, size1=2, size2=(15, 15), size3=(19, 19), size4=(19, 19)):
    # compute geometric mean between image and its offsets
    # the mean overweights the first image: sqrt(sqrt(a * b) * c))
    # mean(mean(a, b), c)) != mean(a, b, c)
    # other possibility: apply log, convolve, apply exp
    blor = image.copy()
    for x in range(size1):
        for y in range(size1):
            blor[x:x - size1, y:y - size1] = np.sqrt(np.abs(blor[x:x - size1, y:y - size1] * image[:-size1, :-size1]))

    # ret = np.nan_to_num(ret).min(axis=2)
    # ret = (ret, ret, ret)
    # ret = np.moveaxis(ret, 0, 2)
    blur_blor = cv2.GaussianBlur(blor, size2, 0)

    luminosity = (np.max(blur_blor, axis=2) + np.min(blur_blor, axis=2)) * 0.5
    saturation = (np.max(blur_blor, axis=2) - np.min(blur_blor, axis=2)) / (1 - np.abs(luminosity * 2 - 1))

    blur_saturation = cv2.GaussianBlur(saturation, size3, 0)
    mix = np.moveaxis((blur_saturation, blur_saturation, blur_saturation), 0, 2) * image
    blur_mix = cv2.GaussianBlur(mix, size4, 0)

    return blur_mix


def main(
        draw_boxes=True,
        draw_ourboxes=True,
        show=False,
        dataset='val',  # 'val' or 'test'
        skip=0,
):
    predictions = {}
    ground_truth = {}
    image_sets = get_image_sets(sets=100, filter=dataset)

    for i in range(skip):
        next(image_sets)

    for images, paths, boxes in image_sets:
        p = Paths.output / paths[3, 4]
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            heatmaps = np.load(str(p.parent / 'heatmap.npy'))
        except:
            heatmaps = get_temporal_diff_heatmaps(images, boxes, blur_color_diff=lambda x: blor(x))
            np.save(str(p.parent / 'heatmap.npy'), heatmaps)
        detection_map, ourboxes = get_detection_map(heatmaps, images, boxes=boxes, thresh_quantile=0.985)
        im = images[3, 4].copy()
        im[np.where(detection_map > 0)] = [0, 0, 255]
        if draw_boxes:
            draw_bounding_boxes(im, boxes)
        if draw_ourboxes:
            draw_bounding_boxes(im, ourboxes, box_color=(255, 255, 0))
        cv2.imwrite(str(p), im)
        print(f'Image exported: {p}')
        if show:
            imshow(im)

        predictions[p.parent.name] = ourboxes.tolist()
        ground_truth[p.parent.name] = boxes.tolist()

    p = Paths.output / f'{dataset}.json'
    with open(p, 'w') as outfile:
        json.dump(predictions, outfile)
    print(f'JSON created: {p}')

    if len(ground_truth):
        ap = compute_AP(predictions, ground_truth)
        print(f"Average precision={ap:.5f} on {dataset} set.")


if __name__ == '__main__':
    main(dataset='val')
    main(dataset='test')
