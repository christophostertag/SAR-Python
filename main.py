###
from glob import glob

import cv2
import numpy as np
from matplotlib import pyplot as plt

from common import get_image_metas, merge_images, select_bounding_boxes, draw_bounding_boxes
from config import Paths
from pixel_clustering.common import load_image_batches, load_bounding_boxes, crop190, load_mask

# image_batches = load_image_batches(path=image_path, batch_size=n_images, filter=data)
# bounding_boxes = load_bounding_boxes()
# images, rel_image_paths = next(image_batches)
# rel_image_paths = [p.relative_to(image_path) for p in rel_image_paths]
#
# # mask and crop
# image_mask = crop190(load_mask())
# images = crop190(images) * image_mask

folders = Paths.data_validation.glob('*-*/')
image_paths, homographies = get_image_metas(folders)
image_mask = cv2.imread(str(Paths.data / 'mask.png'), 0)

for folder_index, folder in enumerate(image_paths):
    merged_images = []

    for timestamp_index, timestamp in enumerate(folder):
        wraped_images = []

        for image_path in timestamp:
            image = cv2.imread(str(image_path))
            masked_image = cv2.bitwise_and(image, image, mask=image_mask)  # mask

            width, height, _ = masked_image.shape
            image_name = image_path.stem
            M = np.array(homographies[folder_index][image_name])

            warped_image = cv2.warpPerspective(masked_image, M, (width, height))
            wraped_images.append(warped_image)

            # plt.imshow(warped_image)
            # plt.show()

        wraped_images = np.array(wraped_images)

        merged_image = merge_images(wraped_images)
        merged_images.append(merged_image)

        cv2.imwrite(str(Paths.data / 'wraped' / f'{folder_index}-{timestamp_index}.png'), merged_image)
        # plt.imshow(merged_image)
        # plt.show()

    merged_images = np.array(merged_images)

    fully_merged_image = merge_images(merged_images)

    # main_image = merged_images[3]
    #
    # # main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    # # fully_merged_image = cv2.cvtColor(fully_merged_image, cv2.COLOR_BGR2GRAY)
    #
    # # main_image = main_image[190:-190, :]
    # # fully_merged_image = fully_merged_image[190:-190, :]
    #
    # bounding_boxes = load_bounding_boxes()
    # boxes = select_bounding_boxes(image_paths[0][3][0], bounding_boxes)
    #
    #
    # # width, height = image_mask.shape
    # # image_name = 'mask'
    # # M = np.array(homographies[0]['3-G03'])
    # #
    # # warped_mask = cv2.warpPerspective(image_mask, M, (width, height))
    # #
    # # plt.imshow(warped_mask)
    # # plt.show()
    #
    # # plt.imshow(main_image - fully_merged_image)
    # # plt.show()
    # # plt.imshow(-(main_image - fully_merged_image))
    # # plt.show()
    # ksize = (5, 5)
    # plt.imshow(cv2.GaussianBlur(main_image, ksize,0) )
    # plt.show()
    # # # plt.imshow(-(cv2.blur(main_image, ksize) - cv2.blur(fully_merged_image, ksize)))
    # # # plt.show()
    # # # plt.imshow(cv2.blur(main_image - fully_merged_image, ksize))
    # # # plt.show()
    # # # plt.imshow(-cv2.blur(main_image - fully_merged_image, ksize))
    # # # plt.show()
    # # plt.imshow(-cv2.bitwise_and(main_image, fully_merged_image))
    # # plt.show()
    #
    # cut = 470
    # mask = np.zeros((1024,1024), np.uint8)
    # mask[cut:-1-cut, cut:-1-cut] = 1
    # img = np.zeros_like(main_image)
    #
    # f = np.fft.fft2(main_image[:,:,0])
    # fshift = np.fft.fftshift(f)
    # fshift = fshift*mask
    # f_ishift = np.fft.ifftshift(fshift)
    # img_back = np.fft.ifft2(f_ishift)
    # img[:,:,0] = np.real(img_back)
    # magnitude_spectrum = 20 * np.log(np.abs(fshift))
    # plt.imshow(magnitude_spectrum, cmap='gray')
    # plt.show()
    #
    # f = np.fft.fft2(main_image[:, :, 1])
    # fshift = np.fft.fftshift(f)
    # fshift = fshift * mask
    # f_ishift = np.fft.ifftshift(fshift)
    # img_back = np.fft.ifft2(f_ishift)
    # img[:, :, 1] = np.real(img_back)
    # magnitude_spectrum = 20 * np.log(np.abs(fshift))
    # plt.imshow(magnitude_spectrum, cmap='gray')
    # plt.show()
    #
    # f = np.fft.fft2(main_image[:, :, 2])
    # fshift = np.fft.fftshift(f)
    # fshift = fshift * mask
    # f_ishift = np.fft.ifftshift(fshift)
    # img_back = np.fft.ifft2(f_ishift)
    # img[:, :, 2] = np.real(img_back)
    # magnitude_spectrum = 20 * np.log(np.abs(fshift))
    # plt.imshow(magnitude_spectrum, cmap='gray')
    # plt.show()
    # plt.imshow(img)
    # plt.show()
    #
    # draw_bounding_boxes(main_image, boxes, x_offset=0)
    # draw_bounding_boxes(fully_merged_image, boxes, x_offset=0)
    #
    # plt.imshow(main_image)
    # plt.show()
    # plt.imshow(fully_merged_image)
    # plt.show()
    break