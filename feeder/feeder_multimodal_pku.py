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
            #transforms.Resize(size=224),
            transforms.Resize(size=225),
            #transforms.ColorJitter(hue=.05, saturation=.05),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(20, resample=Image.BILINEAR),
            #transforms.RandomErasing(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_evaluation = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize(size=224),
            transforms.Resize(size=225),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_weight = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=225),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        # add RGB features based on self.sample_name  -- B
        # print('self.sample_name',self.sample_name)
        if self.random_interval:
            rgb = rgb_roi.construct_st_roi(self.sample_name[index][0:16], self.evaluation, self.random_interval,self.random_roi_move,self.random_flip)
        else:
            rgb = self.sample_name[index][0:16] + '.png'
            rgb = Image.open(self.rgb_path + rgb)
        rgb = np.array(rgb.getdata())

        rgb = torch.from_numpy(rgb).float()
        T, C = rgb.size()

        rgb = rgb.permute(1, 0).contiguous()
        rgb = rgb.view(C, 480, 480)
        '''
        rgb = rgb.numpy()
        rgb = self.preprocess(rgb)
        rgb = torch.from_numpy(rgb)
        '''
        if self.evaluation:
            rgb = self.transform_evaluation(rgb)
        else:
            rgb = self.transform(rgb) # resize to 224x224

        '''
        # add manual weight
        weight = self.attentionmask(label)
        weight = self.transform_weight(torch.from_numpy(weight).float())
        '''
        #'''
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
        #'''


        return data_numpy, rgb, label#, weight
        #return rgb, label, weight

    def preprocess(self, img_data):
        #mean_vec = np.array([0.485, 0.456, 0.406]) # ImageNet
        #stddev_vec = np.array([0.229, 0.224, 0.225]) # ImageNet
        # calculate meand and std: https://discuss.pytorch.org/t/get-value-out-of-torch-cuda-float-tensor/2539/5
        #tensor([106.6353, 101.6921,  87.4993], device='cuda:0')
        #tensor([58.0973, 58.3108, 61.3052], device='cuda:0')
        #mean_vec = np.array([0.4182, 0.3988, 0.3431]) # Testing set
        #stddev_vec = np.array([0.2278, 0.2287, 0.2404]) # Testing set
        #mean_vec = np.array([0.3983, 0.3791, 0.3177]) # training set size 162 without gussian mask
        #stddev_vec = np.array([0.2350, 0.2295, 0.2356]) # training set size 162 without gussian mask
        #mean_vec = np.array([0.3193, 0.3110, 0.2567]) # training set size 162 without gussian mask
        #stddev_vec = np.array([0.2483, 0.2427, 0.2460])
        #mean_vec = np.array([0.3193, 0.3110, 0.2567]) # training set size 162 without gussian mask
        #stddev_vec = np.array([0.2483, 0.2427, 0.2460])
        mean_vec = np.array([0, 0, 0])
        stddev_vec = np.array([1, 1, 1])
        norm_img_data = np.zeros(img_data.shape).astype('float32')
        for i in range(img_data.shape[0]):
             # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
             #norm_img_data[i,:,:] = np.multiply(norm_img_data[i,:,:], self.gussianmask)
             norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]

        return norm_img_data

    def gussianmask(self, kernel_size=81, sigma=9):
        #kernel_size = 81
        #sigma = 15

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp((
                              -torch.sum((xy_grid - mean)**2., dim=-1) /\
                              (2*variance)).float()
                          )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # normalize to [0, 1]
        gaussian_kernel_nor = gaussian_kernel - gaussian_kernel.min()
        gaussian_kernel_nor = gaussian_kernel_nor / (gaussian_kernel_nor.max() - gaussian_kernel.min())
        gaussian_kernel_nor = gaussian_kernel_nor.mul(255).byte()
        gaussian_kernel_nor_ = torch.cat((gaussian_kernel_nor, gaussian_kernel_nor), 1)
        gaussian_kernel_nor_ = torch.cat((gaussian_kernel_nor_, gaussian_kernel_nor_), 0)
        #gaussian_kernel_nor_ = horzcat(gaussian_kernel_nor_, gaussian_kernel_nor_)
        gaussian_kernel_nor_ = gaussian_kernel_nor_.numpy()
        return gaussian_kernel_nor_

    def attentionmask(self, action):
        att_mask = np.full((3, 480, 480),0)

        if action in [35]: # 1
            att_mask[:, 0:96, :]=1.0
        elif action in [6, 9, 10, 11, 12, 24, 28, 29, 30, 31, 33, 44, 45]: # 23
            att_mask[:, 96:288, :]=1.0
        elif action in [23,25]: # 45
            att_mask[:, 288:480, :]=1.0
        elif action in [0, 1, 2, 3, 7, 8, 13, 14, 17, 18, 19, 20, 21, 22, 27, 32, 34, 36, 37, 38, 39, 40, 43, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57]: # 123
            att_mask[:, 0:288, :]=1.0
        elif action in [50]: # 145
            att_mask[:, 0:96, :]=1.0
            att_mask[:, 288:480, :]=1.0
        elif action in [4, 5, 15, 16, 41]: # 2345
            att_mask[:, 96:480, :]=1.0
        elif action in [26, 42, 58, 59]: # 12345
            att_mask[:, :, :]=1.0
        else:
            print('ERROR: action not found')

        return att_mask
