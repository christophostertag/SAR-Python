from glob import glob

import cv2
import matplotlib.pyplot as plt  # plotting library
import numpy as np  # this module is useful to work with numerical arrays
import pandas as pd
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import json
import models
from common import imshow
from temporal_diff.temporal_diff import get_temporal_diff_heatmaps, get_detection_map
from util import add_noise, ImageDataset, WrapedImageDataset, datadir
from torchinfo import summary
from common import load_bounding_boxes, draw_bounding_boxes
from pathlib import Path


class AnomalyDetector():

    def __init__(self, num_channels=3, channels_per_compresion=8, depths=[3, 4, 5, 6], kernel_size=3):

        self.AEs = [
            models.Autoencoder(num_channels=num_channels, channels_per_compresion=channels_per_compresion, depth=depth,
                               kernel_size=kernel_size) for depth in depths]

    def __call__(self, x_list, epoch):
        anomaly_map_list = []
        diffs_list = []
        for x_index, x in enumerate(x_list):
            anomaly_map = np.zeros(x.shape[1:])
            npx = x.cpu().squeeze().numpy()

            # cv2.imwrite(f'output/IMG{x_index}_origin_EPOCH{epoch+1}.png', cv2.cvtColor(np.swapaxes(npx*256,0,2), cv2.COLOR_RGB2BGR))
            diffs = []
            for i, AE in enumerate(self.AEs):
                AE.eval()
                with torch.no_grad():
                    reconstruction = AE(x).cpu().squeeze().numpy()

                diff = np.abs(npx - reconstruction)
                # diff = cv2.GaussianBlur(diff, (7,7),0)
                # diff = diff / np.max(diff)
                # diff = np.swapaxes(diff,0,2)
                # diff = np.swapaxes(diff,0,2)
                # size= 32
                # mean = np.zeros_like(diff)
                # for xi in range(0, 1024, size):
                #     for y in range(0, 1024, size):
                #         mean[:, xi:xi + size, y:y + size] = np.mean(diff[:,xi:xi+size,y:y+size], axis=(1,2), keepdims=True)
                #
                #         # mean = np.mean(diff[2,xi:xi+size,y:y+size], keepdims=True)
                #         # diff[2, xi:xi + size, y:y + size] = mean
                #
                # diff = np.abs(diff - mean)
                # diff = np.swapaxes(diff,0,2)
                # diff = cv2.cvtColor(diff, cv2.COLOR_RGB2HSV)
                # diff = cv2.cvtColor(diff, cv2.COLOR_HSV2RGB)
                # diff = np.swapaxes(diff,0,2)
                #
                # mask = diff / np.max(diff) > 0.25
                # diff = np.zeros_like(diff)
                # diff[mask] = 1
                diffs.append(diff)
                anomaly_map += diff / diff.sum(dtype=np.single)

                # cv2.imwrite(f'output/IMG{x_index}_difference_{i}_EPOCH{epoch+1}.png', cv2.cvtColor(np.swapaxes(diff/ np.max(diff)*256,0,2), cv2.COLOR_RGB2BGR))

            anomaly_map = anomaly_map / np.max(anomaly_map)

            anomaly_map_list.append(anomaly_map)
            diffs_list.append(diffs)

            # anomaly_map32 = np.zeros_like(anomaly_map, dtype=np.single)
            # anomaly_map32 += anomaly_map
            # cv2.imwrite(f'output/IMG{x_index}_anomaly_EPOCH{epoch+1}.png', cv2.cvtColor(np.swapaxes((anomaly_map32*256),0,2), cv2.COLOR_RGB2BGR))

        return anomaly_map_list, diffs_list

    def to(self, device):
        for AE in self.AEs:
            AE.to(device)

    def load(self, file_names):
        for i, AE in enumerate(self.AEs):
            AE.load_state_dict(torch.load(file_names[i]))

    def summarize(self):
        for AE in self.AEs:
            summary(AE, input_size=(batch_size, 3, 1023, 1023))

    def train_epoch(self, device, dataloader, validation_dataloader, loss_fn, optimizer, noise_factor=0.5):
        mean_loss = []
        for i, AE in enumerate(self.AEs):
            # Set train mode for both the encoder and the decoder
            AE.train()
            train_loss = []
            # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
            for image_batch, _ in dataloader:
                # Move tensor to the proper device
                image_noisy = add_noise(image_batch, noise_factor)
                image_batch = image_batch.to(device)
                image_noisy = image_noisy.to(device)
                # Encode data
                reconstuction = AE(image_noisy)
                # Evaluate loss
                loss = loss_fn(reconstuction, image_batch)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Print batch loss
                print(f'\t AE{i}  :  partial train loss (single batch): {loss.data}')
                train_loss.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                AE.eval()
                val_loss = []
                for image_batch, _ in validation_dataloader:
                    # Move tensor to the proper device
                    image_noisy = add_noise(image_batch, noise_factor)
                    image_batch = image_batch.to(device)
                    image_noisy = image_noisy.to(device)
                    # Encode data
                    reconstuction = AE(image_noisy)
                    # Evaluate loss
                    loss = loss_fn(reconstuction, image_batch)
                    # Print batch loss
                    print(f'\t AE{i}  :  partial validation loss (single batch): {loss.data}')
                    val_loss.append(loss.detach().cpu().numpy())

            mean_loss.append({'train': np.mean(train_loss), 'valid': np.mean(val_loss)})

        return mean_loss

    def plot_ae_outputs(self, epoch, n=5):
        plt.figure(figsize=(12 * n, 12 * (len(self.AEs) + 1)))
        for i in range(n):
            ax = plt.subplot((len(self.AEs) + 1), n, i + 1)
            img = validation_dataset[i][0].unsqueeze(0).to(device)

            plt.imshow(np.swapaxes(img.cpu().squeeze().numpy(), 0, 2))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n // 2:
                ax.set_title('Original images')

            for j, AE in enumerate(self.AEs):
                AE.eval()
                with torch.no_grad():
                    reconstruction = AE(img)

                ax = plt.subplot((len(self.AEs) + 1), n, i + 1 + n * (j + 1))
                plt.imshow(np.swapaxes(reconstruction.cpu().squeeze().numpy(), 0, 2))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == n // 2:
                    ax.set_title('Reconstructed images')
        plt.savefig(f'output/EPOCH{epoch + 1}_reconstructions.png')


if __name__ == '__main__':

    train_transform = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = ImageDataset('train_warped', transform=train_transform)
    validation_dataset = ImageDataset('validation_warped_time_aligned', transform=test_transform)
    test_dataset = ImageDataset('test_warped', transform=test_transform)

    m = len(train_dataset)

    batch_size = 30

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=3, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, num_workers=3)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=3)

    loss_fn = torch.nn.MSELoss(reduction='sum')

    ### Define an optimizer (both for the encoder and the decoder!)
    lr = 0.0001

    ### Set the random seed for reproducible results
    torch.manual_seed(0)

    model = AnomalyDetector(num_channels=3, channels_per_compresion=8, depths=[2, 3, 4, 5], kernel_size=5)

    params_to_optimize = [{'params': AE.parameters()} for AE in model.AEs]

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    model.to(device)

    model.summarize()

    if False:
        start_epoch = 16
        end_epoch = 300

        if start_epoch != 0:
            model.load(glob(f'output/EPOCH{start_epoch}*'))

        img = []
        for i in range(7):
            img.append(validation_dataset[random.randint(0, len(validation_dataset))][0].unsqueeze(0).to(device))

        diz_loss = {'train': [], 'valid': []}
        for epoch in range(start_epoch, end_epoch):
            loss = model.train_epoch(device, train_loader, valid_loader, loss_fn, optim)

            for i, AE in enumerate(model.AEs):
                torch.save(AE.state_dict(), f'output/EPOCH{epoch + 1}_AE{i}_val_loss_{loss[i]["valid"]}_weights.pth')

            print(f'\n EPOCH {epoch + 1}/{end_epoch} \t  loss : {loss} \t ')
            diz_loss['train'].append([loss[i]['train'] for i in range(len(model.AEs))])
            diz_loss['valid'].append([loss[i]['valid'] for i in range(len(model.AEs))])

            model.plot_ae_outputs(epoch, n=5)

            model(img, epoch)

            plt.figure(figsize=(16, 16))
            for i in range(len(model.AEs)):
                plt.plot(range(start_epoch + 1, epoch + 2), np.array(diz_loss['train'])[:, i],
                         label='train loss: AE' + str(i), color=plt.cm.RdYlBu(i * 50))
                plt.plot(range(start_epoch + 1, epoch + 2), np.array(diz_loss['valid'])[:, i], '--',
                         label='validation loss: AE' + str(i), color=plt.cm.RdYlBu(i * 50))
            plt.legend()
            plt.savefig("output/Loss.png")
    else:
        model.load(glob('output/EPOCH4_*'))

    img = []
    for i in range(7):
        img.append(validation_dataset[i][0].unsqueeze(0).to(device))

    # plt.figure(figsize=(16, 16))
    # ax = plt.subplot(1, 1, 1)
    # plt.imshow(similarity(img), cmap='gray')
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # ax.set_title('Similarity Map')
    # plt.show()

    with open(datadir + '/validation/labels.json', 'r') as file:
        all_bounding_boxes = json.load(file)
    subdirs = list(Path(datadir + '/validation').glob('valid*'))
    model = AnomalyDetector(num_channels=3, channels_per_compresion=8, depths=[2,3,5], kernel_size=5)
    validation_dataset.transform = test_transform
    valid_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=7, num_workers=3)
    epochs = range(1, 31)
    for ae0 in range(9,18):
        for ae1 in list(range(12, 16)) + list(range(16, 31, 2)):
            for ae2 in list(range(3, 11)) + list(range(12, 31, 3)):
                model.load(glob(f'output/EPOCH{ae0}_AE0*') + glob(f'output/EPOCH{ae1}_AE1*') + glob(f'output/EPOCH{ae2}_AE3*'))

                for i, (images, _) in enumerate(valid_loader):
                    anomaly_maps, difference_maps = model(images[:, None], 0)

                    for image_index in range(len(anomaly_maps)):
                        anomaly_map32 = np.zeros_like(anomaly_maps[image_index], dtype=np.single)
                        anomaly_map32 += anomaly_maps[image_index]
                        anomaly_map32 = anomaly_map32 * 256
                        anomaly_map32 = np.swapaxes(anomaly_map32, 0, 2)
                        anomaly_map32 = np.swapaxes(anomaly_map32, 0, 1)
                        draw_bounding_boxes(anomaly_map32, all_bounding_boxes[subdirs[i].name], box_color=(255, 170, 0))
                        cv2.imwrite(f'output/compare/IMG{i}_FRAME{image_index}_anomaly_config[{ae0}, {ae1}, {ae2}].png',
                                    cv2.cvtColor(anomaly_map32, cv2.COLOR_RGB2BGR))


