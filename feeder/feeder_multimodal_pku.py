# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from feeder import segment_rgbbody_5fs_pkummd as rgb_roi

# visualization
import time
import math
# operation
from . import tools

# rgb --B
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 random_flip=False,
                 random_interval=False,
                 random_roi_move=False,
                 centralization=False,
                 window_size=-1,
                 debug=False,
                 evaluation=False,
                 mmap=True):
        self.debug = debug
        self.evaluation = evaluation
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.random_flip = random_flip
        self.random_roi_move = random_roi_move
        self.random_interval = random_interval
        self.centralization = centralization

        self.rgb_path = '/media/bruce/2Tssd/data/pku_rgb_frames_crop/fivefs/'
        self.load_data(mmap)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=225),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.gussianmask = self.gussianmask()/255

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        label = self.label[index]

        if self.evaluation:
            rgb = self.sample_name[index][0:16] + '.png'
            rgb = Image.open(self.rgb_path + rgb)
        else:
            rgb = rgb_roi.construct_st_roi(self.sample_name[index][0:16], self.evaluation, self.random_interval,self.random_roi_move,self.random_flip)

        rgb = np.array(rgb.getdata())

        rgb = torch.from_numpy(rgb).float()
        T, C = rgb.size()

        rgb = rgb.permute(1, 0).contiguous()
        rgb = rgb.view(C, 480, 480)

        rgb = self.transform(rgb)

        # get data
        data_numpy = np.array(self.data[index])
        if self.centralization:
            data_numpy = tools.centralization(data_numpy)
        # processing
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, rgb, label#, weight
