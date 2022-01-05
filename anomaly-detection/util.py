from glob import glob

import numpy as np
import torch
from  torch.utils.data import Dataset
import cv2


def add_noise(tensor, noise_factor):
    return tensor + torch.randn(tensor.size()) * noise_factor


def similarity(tensor : np.ndarray, kernel_size=3):
    similarity_map = np.zeros_like(tensor[:-kernel_size+1,:-kernel_size+1,0])
    for x in range(similarity_map.shape[0]):
        print(x)
        for y in range(similarity_map.shape[1]):
            norm = np.linalg.norm(tensor[x + int(kernel_size/2), y + int(kernel_size/2),:])
            for kernelx in range(kernel_size):
                for kernely in range(kernel_size):
                    if kernelx != int(kernel_size/2) or kernely != int(kernel_size/2):
                        similarity_map[x,y] += np.dot(tensor[x + kernelx, y + kernely,:], tensor[x + int(kernel_size/2), y + int(kernel_size/2),:]) / (np.linalg.norm(tensor[x + kernelx, y + kernely,:]) * norm)

    return similarity_map / (kernel_size**2 - 1)


class ImageDataset(Dataset):
    def __init__(self, dataset, colorspace=cv2.COLOR_BGR2HSV, transform=None, target_transform=None):
        self.images = sorted(glob('../data/' + dataset +'/*/*.png', recursive=False))
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
            image = self.transform(image)
        if self.target_transform:
            pass #label = self.target_transform(label)
        return image, 1#, label

    class WarpedImageDataset(Dataset):
        def __init__(self, dataset, data_dir='../data/', colorspace=cv2.COLOR_BGR2RGB, transform=None, target_transform=None):

            self.mask = cv2.imread(data_dir + 'mask.png', 0)
            self.images = sorted(glob(data_dir + dataset + '/*/*.png', recursive=False))
            self.colorspace = colorspace
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = self.images[idx]
            image = cv2.cvtColor(cv2.imread(img_path), self.colorspace)[1:-2, 1:-2]
            # label = cv2.cvtColor(cv2.imread(img_path), self.colorspace)[:-1,:-1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                pass  # label = self.target_transform(label)
            return image, 1  # , label