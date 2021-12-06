from glob import glob

import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import models
from util import add_noise, ImageDataset, similarity
from torchinfo import summary





class AnomalyDetector():

    def __init__(self, num_channels=3, channels_per_compresion=8, depths=[3,4,5,6], kernel_size=3):

        self.AEs = [models.Autoencoder(num_channels=num_channels, channels_per_compresion=channels_per_compresion, depth=depth, kernel_size=kernel_size) for depth in depths]


    def __call__(self, x):
        anomaly_map = np.zeros(x.shape[1:])
        npx = x.cpu().squeeze().numpy()

        plt.figure(figsize=(16, 16))
        ax = plt.subplot(1,1,1)
        plt.imshow(np.swapaxes(npx, 0, 2))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Original image')
        plt.show()

        for i, AE in enumerate(self.AEs):
            AE.eval()
            with torch.no_grad():
                reconstruction = AE(x).cpu().squeeze().numpy()

            diff = np.abs(npx - reconstruction)
            anomaly_map += diff * diff.sum()

            plt.figure(figsize=(16, 16))
            ax = plt.subplot(1,1,1)
            plt.imshow(np.swapaxes(diff/ np.max(diff), 0, 2))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title('Difference Map '+str(i))
            plt.show()

        anomaly_map = anomaly_map / np.max(anomaly_map)


        plt.figure(figsize=(16, 16))
        ax = plt.subplot(1,1,1)
        plt.imshow(np.swapaxes(anomaly_map, 0, 2))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Anomaly Map')

        plt.show()

        return anomaly_map

    def to(self, device):
        for AE in self.AEs:
            AE.to(device)

    def load(self, file_names):
        for i, AE in enumerate(self.AEs):
            AE.load_state_dict(torch.load(file_names[i]))

    def summarize(self):
        for AE in self.AEs:
            summary(AE, input_size=(batch_size, 3, 1023, 1023))

    def train_epoch(self, device, dataloader, loss_fn, optimizer, noise_factor=0.3):
        mean_loss = []
        for i, AE in enumerate(self.AEs):
            # Set train mode for both the encoder and the decoder
            AE.train()
            train_loss = []
            # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
            for image_batch, _ in dataloader:  # with "_" we just ignore the labels (the second element of the dataloader tuple)
                # Move tensor to the proper device
                image_noisy = add_noise(image_batch, noise_factor)
                image_noisy = image_noisy.to(device)
                # Encode data
                reconstuction = AE(image_noisy)
                # Evaluate loss
                loss = loss_fn(reconstuction, image_noisy)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Print batch loss
                print('\t AE %d  :  partial train loss (single batch): %f' % (i, loss.data))
                train_loss.append(loss.detach().cpu().numpy())

            mean_loss.append(np.mean(train_loss))

        return mean_loss

    def plot_ae_outputs(self, n=5):
        plt.figure(figsize=(4 * n, 4 * (len(self.AEs) + 1)))
        for i in range(n):
            ax = plt.subplot((len(self.AEs) + 1), n, i + 1)
            img = test_dataset[i][0].unsqueeze(0).to(device)

            plt.imshow(np.swapaxes(img.cpu().squeeze().numpy(), 0, 2))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n // 2:
                ax.set_title('Original images')

            for j, AE in enumerate(self.AEs):
                AE.eval()
                with torch.no_grad():
                    reconstruction = AE(img)

                ax = plt.subplot((len(self.AEs) + 1), n, i + 1 + n * (j+1))
                plt.imshow(np.swapaxes(reconstruction.cpu().squeeze().numpy(), 0, 2))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == n // 2:
                    ax.set_title('Reconstructed images')
        plt.show()





train_dataset = ImageDataset('train')
validation_dataset = ImageDataset('validation')
test_dataset = ImageDataset('test')

train_transform = transforms.Compose([
transforms.ToTensor(),
])

test_transform = transforms.Compose([
transforms.ToTensor(),
])

train_dataset.transform = train_transform
validation_dataset.transform = train_transform
test_dataset.transform = test_transform

m=len(train_dataset)

batch_size=16

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

loss_fn = torch.nn.MSELoss()

### Define an optimizer (both for the encoder and the decoder!)
lr= 0.001

### Set the random seed for reproducible results
torch.manual_seed(0)


model = AnomalyDetector(num_channels=3, channels_per_compresion=8, depths=[2,3,4,5], kernel_size=5)

params_to_optimize = [ {'params': AE.parameters()} for AE in model.AEs]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
model.to(device)

model.summarize()

if False:
    num_epochs = 30
    diz_loss = {'train_loss':[],'val_loss':[]}
    for epoch in range(num_epochs):
        train_loss =model.train_epoch(device, train_loader,loss_fn,optim)

        for i, AE in enumerate(model.AEs):
            torch.save(AE.state_dict(), 'EPOCH{}_AE{}_train_loss_{}_weights.pth '.format(epoch + 1, i,train_loss))

        print('\n EPOCH {}/{} \t train loss {} \t '.format(epoch + 1, num_epochs,train_loss))
        diz_loss['train_loss'].append(train_loss)

        model.plot_ae_outputs(n=5)
else:
    model.load(glob('EPOCH60*'))


for i in range(4):
    img = validation_dataset[i*6][0].unsqueeze(0).to(device)

    # plt.figure(figsize=(16, 16))
    # ax = plt.subplot(1, 1, 1)
    # plt.imshow(similarity(img), cmap='gray')
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # ax.set_title('Similarity Map')
    # plt.show()

    model(img)