import json
import random
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from  torch.utils.data import Dataset
import cv2


datadir = 'E:/Dokumente/data'

def add_noise(tensor, noise_factor):
    return tensor + torch.randn(tensor.size()) * noise_factor


def merge_images(images : np.ndarray, method='mean'):
    if method is 'mean':
        merged = np.mean(images, axis=0)

    if method is 'and':
        merged = images[0]
        for image in images[1:]:
            merged = cv2.bitwise_and(merged, image)

    return merged


class ImageDataset(Dataset):
    def __init__(self, dataset, colorspace=cv2.COLOR_BGR2RGB, transform=None, target_transform=None):
        self.images = sorted(glob(datadir + '/' + dataset +'/*/*.png', recursive=False))
        self.colorspace = colorspace
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.cvtColor(cv2.imread(img_path), self.colorspace)[1:-2,1:-2]
        #label = cv2.cvtColor(cv2.imread(img_path), self.colorspace)[:-1,:-1]
        if self.transform:
            image = self.transform(Image.fromarray(image))
        if self.target_transform:
            pass #label = self.target_transform(label)
        return image, 1#, label

class WrapedImageDataset(Dataset):
    def __init__(self, dataset, time_aligned=False, colorspace=cv2.COLOR_BGR2RGB, transform=None, target_transform=None):
        self.path = datadir + '/' + dataset
        self.timesteps = range(7)
        self.image_mask = cv2.imread(self.path + '/../mask.png', 0)
        self.homographies = []
        for folder in glob(self.path + '/*/', recursive=False):
            self.homographies.append(json.loads(open(folder + '/homographies.json', 'r').read()))
        self.time_aligned = time_aligned
        self.restock()
        self.colorspace = colorspace
        self.transform = transform
        self.target_transform = target_transform

    def restock(self):
        folders = glob(self.path + '/*/', recursive=False)
        self.images = []
        for folder in folders:
            folder_images = []
            for timestep in self.timesteps:
                folder_images.append(glob(folder + f'/{timestep}*.png'))
            self.images.append(folder_images)
        self.folders = len(folders)

    def __len__(self):
        return self.folders * len(self.timesteps)

    def __getitem__(self, idx):
        # if sum(self.folders) == 0:
        #     self.restock()

        folder_index = idx % self.folders

        wraped_masks = []
        wraped_images = []
        image_index = 0
        if self.time_aligned:
            timestep = [int(idx / self.folders)] * 10
        else:
            timestep = list(self.timesteps) + np.random.choice(self.timesteps, 3).tolist()

        for image_index in range(len(self.images[folder_index][timestep[0]])):

            image = cv2.cvtColor(cv2.imread(self.images[folder_index][timestep[image_index]][image_index]), self.colorspace)
            image_name = Path(self.images[folder_index][timestep[image_index]][image_index]).stem

            width, height, _ = image.shape

            M = np.array(self.homographies[folder_index][image_name])

            wraped_images.append(cv2.warpPerspective(image, M, (width, height)))
            wraped_masks.append(cv2.warpPerspective(self.image_mask, M, (width, height)))

        wraped_images = np.array(wraped_images)
        wraped_masks = np.array(wraped_masks)

        merged_image = merge_images(wraped_images)
        merged_mask = merge_images(wraped_masks, method='and')

        masked_image = cv2.bitwise_and(merged_image, merged_image, mask=merged_mask)

        if self.transform:
            image = self.transform(Image.fromarray(masked_image))
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image[:,1:-2,1:-2], 1  # , label


def preprocess(dataset, time_aligned=False):
    path = datadir + '/' + dataset
    outpath = path + '_warped' + ('_time_aligned' if time_aligned else '')
    Path(outpath).mkdir()
    image_mask = cv2.imread(path + '/../mask.png', 0)
    folders = glob(path + '/*/', recursive=False)
    for folder in folders:
        folder_name = Path(folder).stem
        (Path(outpath) / folder_name).mkdir()
        homographies = json.loads(open(folder + '/homographies.json', 'r').read())
        timesteps = range(7)
        folder_images = []
        for timestep in timesteps:
            folder_images.append(glob(folder + f'/{timestep}*.png'))

        if time_aligned:
            timesteps_list = []
            for timestep in timesteps:
                timesteps_list.append([timestep] * 10)
        else:
            timesteps_list = []
            for i in range(300):
                timesteps_list.append(list(timesteps) + random.sample(timesteps, 3))
                random.shuffle(timesteps_list[-1])

        del timesteps

        for timesteps in timesteps_list:

            wraped_masks = []
            wraped_images = []
            for image_index, timestep in enumerate(timesteps):
                image = cv2.imread(folder_images[timestep][image_index])
                image_name = Path(folder_images[timestep][image_index]).stem

                width, height, _ = image.shape

                M = np.array(homographies[image_name])

                wraped_images.append(cv2.warpPerspective(image, M, (width, height)))
                wraped_masks.append(cv2.warpPerspective(image_mask, M, (width, height)))

            wraped_images = np.array(wraped_images)
            wraped_masks = np.array(wraped_masks)

            merged_image = merge_images(wraped_images)
            merged_mask = merge_images(wraped_masks, method='and')

            masked_image = cv2.bitwise_and(merged_image, merged_image, mask=merged_mask)

            cv2.imwrite(str(Path(outpath) / folder_name / f'{timesteps}.png'), masked_image)