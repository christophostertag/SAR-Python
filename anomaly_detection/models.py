import numpy as np
from torch import nn
from torch import sigmoid



class Autoencoder(nn.Module):

    def __init__(self, num_channels=3, channels_per_compresion=8, depth=3, kernel_size=3):
        super().__init__()

        layers = []

        layers.append(nn.Conv2d(in_channels=num_channels, out_channels=channels_per_compresion, kernel_size=kernel_size, stride=2))
        layers.append(nn.ReLU(True))

        for i in range(1, depth):
            layers.append(
                nn.Conv2d(in_channels=channels_per_compresion * (2**(i-1)), out_channels=channels_per_compresion * (2**(i)),
                          kernel_size=kernel_size, stride=2))
            layers.append(nn.ReLU(True))

            if i % 2 == 0 and not depth - i < 3:
                layers.append(nn.BatchNorm2d(channels_per_compresion * (2**(i))))
            if depth - i < 3:
                layers.append(nn.Dropout())

        # layers.append(nn.Flatten())

        self.encoder_cnn = nn.Sequential(*layers)


        # self.linear_block = nn.Sequential(
        #     nn.Linear(channels_per_compresion * (2**(depth-1)) * ((2 ** (10-depth) - 1) ** 2),
        #               channels_per_compresion * (2**(depth-1)) * ((2 ** (10-depth) - 1) ** 2)),
        #     nn.ReLU(True),
        #     nn.Linear(channels_per_compresion * (2**(depth-1)) * ((2 ** (10-depth) - 1) ** 2),
        #               channels_per_compresion * (2**(depth-1)) * ((2 ** (10-depth) - 1) ** 2)),
        #     nn.ReLU(True),
        #     nn.Linear(channels_per_compresion * (2**(depth-1)) * ((2 ** (10-depth) - 1) ** 2),
        #               channels_per_compresion * (2**(depth-1)) * ((2 ** (10-depth) - 1) ** 2)),
        #     nn.ReLU(True),
        # )

        layers = []

        # layers.append(nn.Unflatten())

        for i in range(depth-1, 0, -1):
            layers.append(nn.ConvTranspose2d(in_channels=channels_per_compresion * (2**(i)), out_channels=channels_per_compresion * (2**(i-1)), kernel_size=kernel_size, stride=2))
            layers.append(nn.ReLU(True))


            if i % 2 == 0 and not depth - i < 3:
                layers.append(nn.BatchNorm2d(channels_per_compresion * (2**(i-1))))

        layers.append(nn.ConvTranspose2d(in_channels=channels_per_compresion, out_channels=num_channels, kernel_size=kernel_size, stride=2))

        self.decoder_cnn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.decoder_cnn(x)
        x = sigmoid(x)
        return x


