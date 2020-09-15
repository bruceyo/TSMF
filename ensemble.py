#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:32:11 2020

@author: bruce
"""

import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='pku_xsub', choices={'pku_xsub', 'pku_xview', 'ntu_xview', 'ntu_xsub'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
arg = parser.parse_args()

dataset = arg.datasets
alpha = arg.alpha

label = open('./data/'+ dataset +'/val_label_stgcn.pkl', 'rb')
label = np.array(pickle.load(label))

label_g3d = open('./data/'+ dataset +'/val_label_msg3d.pkl', 'rb')
label_g3d = np.array(pickle.load(label_g3d))

r1_teacher = open('./trained_models/'+dataset+'_teacher.pkl', 'rb')
r1_teacher = list(pickle.load(r1_teacher).items())
r1_student = open('./trained_models/'+dataset+'_student.pkl', 'rb')
r1_student = list(pickle.load(r1_student).items())

r1_2s = open('./trained_models/'+dataset+'_agcn_joint.pkl', 'rb')
r1_2s = list(pickle.load(r1_2s).items())

r1_g3d = open('./trained_models/'+dataset+'_msg3d_joint.pkl', 'rb')
r1_g3d = list(pickle.load(r1_g3d).items())


#'''
label_g3d_name = label_g3d[0,:].tolist()
label_name = label[0,:].tolist()
label_in_g3d = np.zeros((len(label[0]),1),dtype='int')
label_in_2s =  np.zeros((len(label[0]),1),dtype='int')

for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    label_in_g3d[i] = label_g3d_name.index(label[0, i])
for i in tqdm(range(len(label_g3d[0]))):
    _, l = label_g3d[:, i]
    label_in_2s[i] = label_name.index(label_g3d[0, i])

right_num_11_teacher = right_num_11_tsmf = right_num_teacher = right_num_tsmf = 0
right_num_11_2s = right_num_11_2s_rgb = right_num_2s = right_num_2s_rgb = 0
right_num_11_g3d = right_num_11_g3d_rgb = right_num_g3d = right_num_g3d_rgb = 0
total_num = 0
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]

    _, r11_teacher = r1_teacher[i]
    _, r33_student = r1_student[i]

    r_teacher = r11_teacher
    r_tsmf = r11_teacher + alpha*r33_student
    r_teacher = np.argmax(r_teacher)
    r_tsmf = np.argmax(r_tsmf)
    right_num_teacher += int(r_teacher == int(l))
    right_num_tsmf += int(r_tsmf == int(l))

    total_num += 1
    #confusion_matrix[int(l)][r] += 1
    _, r11_2s = r1_2s[i]
    r_2s = r11_2s
    r_2s_rgb = r11_2s + alpha*r33_student
    r_2s = np.argmax(r_2s)
    r_2s_rgb = np.argmax(r_2s_rgb)
    right_num_2s += int(r_2s == int(l))
    right_num_2s_rgb += int(r_2s_rgb == int(l))

    _, r11_g3d = r1_g3d[label_in_g3d[i][0]]
    r_g3d = r11_g3d
    r_g3d_rgb = r11_g3d + alpha*r33_student
    r_g3d = np.argmax(r_g3d)
    r_g3d_rgb = np.argmax(r_g3d_rgb)
    right_num_g3d += int(r_g3d == int(l))
    right_num_g3d_rgb += int(r_g3d_rgb == int(l))

acc_teacher = right_num_teacher / total_num
acc_tsmf = right_num_tsmf / total_num
print('\nST-GCN   Joint: {:0.4f}; Joint+RGB: {:0.4f};'.format(acc_teacher,acc_tsmf))

acc_2s = right_num_2s / total_num
acc_2s_rgb = right_num_2s_rgb / total_num
print('2s-GCN   Joint: {:0.4f}; Joint+RGB: {:0.4f};'.format(acc_2s,acc_2s_rgb))

acc_g3d = right_num_g3d / total_num
acc_g3d_rgb = right_num_g3d_rgb / total_num
print('MS-G3D   Joint: {:0.4f}; Joint+RGB: {:0.4f};'.format(acc_g3d,acc_g3d_rgb))
