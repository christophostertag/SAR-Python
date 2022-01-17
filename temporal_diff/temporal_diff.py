import json

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
            color_diff_filtered[i] = blur_color_diff(color_diff_filtered[i] / 255)

        cdiff = color_diff_filtered[7] * 2 * 255 + 128
        draw_bounding_boxes(cdiff, boxes)
        imshow(cdiff)

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


def get_detection_map(
        heatmaps: np.ndarray,
        images:np.ndarray,
        kernel_size=19,
        thresh_quantile=0.95,
        thresh_factor=2,
        min_detections=7,
        boxes=[]
) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2

    # conv_heatmap = convolve2d(heatmaps[0, 7], kernel, mode='same')
    # thresh = thresh_factor * np.quantile(conv_heatmap, q=thresh_quantile)
    # detection = conv_heatmap > thresh
    # d = detection * 255
    # d = np.moveaxis((d, d, d), 0, 2) * 2
    # # draw_bounding_boxes(d, boxes)
    # imshow(d[300:])

    # luminance_maps = []
    # for camera_index in range(10):
    #     for timestep_index in range(7):
    #         luminance_maps.append((np.maximum(images[timestep_index, camera_index, :, :, 0], images[timestep_index, camera_index, :, :, 1], images[timestep_index, camera_index, :, :, 2])+np.minimum(images[timestep_index, camera_index, :, :, 0], images[timestep_index, camera_index, :, :, 1], images[timestep_index, camera_index, :, :, 2]))*0.5)
    #
    #     mean_luminance = np.array(luminance_maps[-7:-1]).mean(axis=0)
    #     for image_index in range(-7, 0):
    #         luminance_maps[image_index] = np.clip(luminance_maps[image_index] - mean_luminance, 0, np.max(luminance_maps))
    #
    #         weight_map = np.empty((1024, 1024, 3))
    #         weight_map[:, :, 0] = luminance_maps[image_index]
    #         weight_map[:, :, 1] = weight_map[:, :, 0]
    #         weight_map[:, :, 2] = weight_map[:, :, 0]
    #         draw_bounding_boxes(weight_map, boxes, box_color=(0, 170, 255))
    #         imshow(weight_map)
    #
    #
    #         weight_map[:, :, 0] = luminance_maps[image_index] - (luminance_maps[image_index-1] if image_index > -7 else 0)
    #         weight_map[:, :, 1] = weight_map[:, :, 0]
    #         weight_map[:, :, 2] = weight_map[:, :, 0]
    #         draw_bounding_boxes(weight_map, boxes, box_color=(0, 170, 255))
    #         imshow(weight_map)

    detection_maps = []
    for heatmap in heatmaps.reshape(-1, *heatmaps.shape[2:]):
        conv_heatmap = convolve2d(heatmap, kernel, mode='same')
        thresh = thresh_factor * np.quantile(np.nan_to_num(conv_heatmap), q=thresh_quantile)
        detection = conv_heatmap > thresh
        detection_maps.append(detection)

    detection_maps = np.reshape(detection_maps, heatmaps.shape)
    weight_map = np.empty((1024,1024,3))
    weight_map[:,:,0] = detection_maps.sum(axis=(0, 1))
    weight_map[:, :, 1] = weight_map[:,:,0]
    weight_map[:, :, 2] = weight_map[:,:,0]
    draw_bounding_boxes(weight_map, boxes, box_color=(0, 170, 255))
    imshow(weight_map*255/20)
    detection_map = detection_maps.sum(axis=(0, 1)) >= min_detections

    return detection_map


def blor(image:np.ndarray, size1=2, size2=(15, 15), size3=(19, 19), size4=(19, 19)):
    blor = image.copy()
    for x in range(size1):
        for y in range(size1):
            blor[x:x - size1, y:y - size1] = np.sqrt(np.abs(blor[x:x - size1, y:y - size1] * image[:-size1, :-size1]))

    # ret = np.nan_to_num(ret).min(axis=2)
    # ret = (ret, ret, ret)
    # ret = np.moveaxis(ret, 0, 2)
    blur_blor = cv2.GaussianBlur(blor, size2, 0)

    luminosity = (np.maximum(np.maximum(blur_blor[:, :, 0], blur_blor[:, :, 1]), blur_blor[:, :, 2]) + np.minimum(np.minimum(blur_blor[:, :, 0], blur_blor[:, :, 1]), blur_blor[:, :, 2])) * 0.5
    saturation = (np.maximum(np.maximum(blur_blor[:, :, 0], blur_blor[:, :, 1]), blur_blor[:, :, 2]) - np.minimum(np.minimum(blur_blor[:, :, 0], blur_blor[:, :, 1]), blur_blor[:, :, 2])) / (1 - np.abs(luminosity * 2 - 1))

    blur_saturation = cv2.GaussianBlur(saturation, size3, 0)
    mix = (np.moveaxis((blur_saturation, blur_saturation, blur_saturation), 0, 2) * image)
    blur_mix = cv2.GaussianBlur(mix, size4, 0)

    return blur_mix

def main(
        draw_boxes=True,
        draw_ourboxes=True,
        show=True,
):
    predictions = {}
    image_sets = get_image_sets(sets=100, filter='valid')
    i = 7
    for images, paths, boxes in image_sets:
        # if i != 0:
        #     i -= 1
        #     continue
        heatmaps = get_temporal_diff_heatmaps(images, boxes, blur_color_diff=lambda x: blor(x))
        detection_map = get_detection_map(heatmaps, images, boxes=boxes, thresh_quantile=0.95)
        im = images[3, 4].copy()
        im[np.where(detection_map > 0)] = [0, 0, 255]
        ourboxes = find_boxes(detection_map)
        if draw_boxes:
            draw_bounding_boxes(im, boxes)
        if draw_ourboxes:
            draw_bounding_boxes(im, ourboxes, box_color=(255, 255, 0))
        p = Paths.output / paths[3, 4]
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), im)
        print(f'Image exported: {p}')
        if show:
            imshow(im)

        predictions[p.parent.name] = ourboxes.tolist()
        pass

    with open('predictions.json', 'w') as outfile:
        json.dump(predictions, outfile)


if __name__ == '__main__':
    main(show=True)
