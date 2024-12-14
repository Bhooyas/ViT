import torch
from torch.utils.data import Dataset
import os
from glob import glob
from PIL import Image
import math
from tqdm import tqdm
import h5py
import numpy as np

class TinyImageNet(Dataset):

    def __init__(self, file, transform=None, target_transform=None):
        super(TinyImageNet, self).__init__()

        # Download Link: https://cs231n.stanford.edu/tiny-imagenet-200.zip

        self.transform = transform
        self.target_transform = target_transform

        file = h5py.File(file, "r")
        self.images = np.array(file["images"])
        self.labels = np.array(file["labels"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
