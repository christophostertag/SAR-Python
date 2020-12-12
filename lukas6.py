from tqdm.notebook import tqdm
import os
from glob import glob
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch import nn 
from torchvision.ops import DeformConv2d 
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval


def autopad(k, p=None):  # kernel, padding
    if p is None:  # add padding to keep spatial dimensions
        # kernel can be scalar or tuple
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    def __init__(self, chi, cho, k: int = 1, s: int = 1, 
        p=None, g=1, act: nn.Module = nn.ReLU(), deformable=False):  
        super().__init__()
        if deformable:
            self.conv = DeformConv2d(chi, cho, k, s, autopad(k, p), groups=g, bias=False)
        else:
            self.conv = nn.Conv2d(chi, cho, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(cho)
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, chi, cho, shortcut=True, g=1, e=0.5):  
        super().__init__()
        chh = int(cho * e)
        self.cv1 = Conv(chi, chh, 1, 1)
        self.cv2 = Conv(chh, cho, 3, 1, g=g)
        self.add = shortcut and chi == cho

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    
    def __init__(self, chi, cho, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        chh = int(cho * e)  
        self.cv1 = Conv(chi, chh, 1, 1)
        self.cv2 = nn.Conv2d(chi, chh, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(chh, chh, 1, 1, bias=False)
        self.cv4 = Conv(2 * chh, cho, 1, 1)
        self.bn = nn.BatchNorm2d(2 * chh)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(chh, chh, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).fuse().autoshape()  # for PIL/cv2/np inputs and NMS
    #for k, v in model.named_parameters(): 
    #    print(k)

    # layer 0 to 9 are backbone part
    

    