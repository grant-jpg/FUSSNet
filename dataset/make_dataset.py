# -*- coding: utf-8 -*-
import h5py, os
import torch, cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

from dataset.pancreas import *


class make_data_3d(Dataset):
    def __init__(self, imgs, plab1, mask1, labs, crop_size = (96, 96, 96)):
        self.img = [img.cpu().squeeze().numpy() for img in imgs]
        self.plab1 = [np.squeeze(lab.cpu().numpy()) for lab in plab1]
        self.mask1 = [np.squeeze(mask.cpu().numpy()) for mask in mask1]
        self.lab = [np.squeeze(lab.cpu().numpy()) for lab in labs]
        self.num = len(self.img)
        self.tr_transform = Compose([
            # RandomRotFlip(),
            CenterCrop(crop_size),
            # RandomNoise(),
            ToTensor()
        ])

    def __getitem__(self, idx):
        samples = self.img[idx], self.plab1[idx], self.mask1[idx], self.lab[idx]
        samples = self.tr_transform(samples)
        imgs, plab1, mask1, labs = samples
        return imgs, plab1.long(), mask1.float(), labs.long()

    def __len__(self):
        return self.num
