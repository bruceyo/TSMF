#!/usr/bin/env python
# pylint: disable=W0201
import os
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor

import sys
import matplotlib.pyplot as plt
from PIL import Image
import time
import pickle
from tqdm import tqdm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        print('Load model st-gcn')
        self.model.stgcn = self.io.load_model('net.st_gcn.Model',
                                              **(self.arg.model_args))
        if self.arg.phase != 'test':
            self.model.stgcn = self.io.load_weights(self.model.stgcn, self.arg.teacher_weights,
                                              self.arg.ignore_weights)
        if self.arg.fix_teacher:
            self.model.stgcn.eval()
            print('Load model st-gcn  DONE')
        #self.model.apply(weights_init)

        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                #Over here we want to only update the parameters of the classifier so
                #self.model.module.classifier.parameters(),
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k, phase):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

        # *********************ensemble with skl results***********************start
        if phase == 'eval':
            r1 = open(self.arg.skeleton_joints_pkl, 'rb')
            r1 = list(pickle.load(r1).items())
            result_dict = dict(
                zip(self.data_loader['test'].dataset.sample_name,
                    self.result))
            self.io.save_pkl(result_dict, 'tmp_test_result.pkl')
            r2 = open(os.path.join(self.arg.work_dir,'tmp_test_result.pkl'), 'rb')
            r2 = list(pickle.load(r2).items())
            right_num = total_num = right_num_5 = 0
            instance_num, class_num = self.result.shape
            confusion_matrix = np.zeros([class_num, class_num])
            for i,l in enumerate(self.label):
                #_, l = label[i]
                _, r11 = r1[i]
                _, r22 = r2[i]
                r = r11 + r22
                rank_5 = r.argsort()[-5:]
                right_num_5 += int(int(l) in rank_5)
                r = np.argmax(r)
                right_num += int(r == int(l))
                total_num += 1
                confusion_matrix[int(l)][r] += 1
            np.savetxt(os.path.join(self.arg.work_dir,'confusion_matrix_epoch_ensemble.csv'), confusion_matrix, fmt='%d', delimiter=",")
            acc = right_num / total_num
            acc5 = right_num_5 / total_num
            self.io.print_log('\t Ensemble Top 1: {:.2f}%; Top 5: {:.2f}%'.format(100 *acc, 100 *acc5))
            accuracy = acc
        # *********************ensemble with skl results***********************end

        if k==1:
            self.progress_info[int(self.meta_info['epoch']/self.arg.eval_interval), 2]  =  100 * accuracy
            if accuracy > self.meta_info['best_t1'] and phase=='eval':
                self.meta_info['best_t1'] = accuracy
                self.meta_info['is_best'] = True
                self.save_recall_precision(self.meta_info['epoch'])
        else:
            self.progress_info[int(self.meta_info['epoch']/self.arg.eval_interval), 3]  =  100 * accuracy

    def save_recall_precision(self, epoch): #original input: (label, score),score refers to self.result
        instance_num, class_num = self.result.shape
        rank = self.result.argsort()
        confusion_matrix = np.zeros([class_num, class_num])

        for i in range(instance_num):
            true_l = self.label[i]
            pred_l = rank[i, -1]
            confusion_matrix[true_l][pred_l] += 1
        np.savetxt(os.path.join(self.arg.work_dir,'confusion_matrix_epoch_{}.csv').format(epoch+1), confusion_matrix, fmt='%d', delimiter=",")

        precision = []
        recall = []

        for i in range(class_num):
            true_p = confusion_matrix[i][i]
            false_n = sum(confusion_matrix[i, :]) - true_p
            false_p = sum(confusion_matrix[:, i]) - true_p
            precision_ = true_p * 1.0 / (true_p + false_p)
            recall_ = true_p * 1.0 / (true_p + false_n)
            if np.isnan(precision_):
                precision_ = 0
            if np.isnan(recall_):
                recall_ = 0
            precision.append(precision_)
            recall.append(recall_)
        recall = np.asarray(recall)
        precision = np.asarray(precision)
        labels = np.asarray(range(1,class_num+1))
        res = np.column_stack([labels.T, recall.T, precision.T])
        np.savetxt(os.path.join(self.arg.work_dir,'recall_precision_epoch_{}.csv'.format(epoch+1)), res, fmt='%.4e', delimiter=",", header="Label,  Recall,  Precision")

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        result_frag = []
        label_frag = []
        loss_value = []
        loss_l1 = []
        loss_l2 = []

        for data, rgb, label in loader:

            data = data.float().to(self.dev)
            label = label.long().to(self.dev)
            rgb = rgb.float().to(self.dev)

            output = self.model(data,rgb)

            '''
            # -------for debug start-------
            # forward for debug
            output, weight, rgb_weighted = self.model(data, rgb)
            #np.einsum('kli->lik', weight.cpu().numpy()[0])
            weight_0 = weight.cpu().numpy()[0,0,:,:]
            weight_0_ = weight_0.astype('float64')
            #weight_0_ = 255.0*weight_0_

            rgb_0_ = np.einsum('kli->lik',rgb[0].cpu().numpy())
            #rgb_weighted = rgb
            rgb_weighted_0 = rgb_weighted[0].cpu().numpy()
            rgb_weighted_0 = np.einsum('kli->lik',rgb_weighted_0)
            plt.figure(figsize=(12, 4))
            plt.subplot(141)
            plt.imshow(weight_0_, cmap='gray', vmin=0, vmax=1)
            plt.subplot(142)
            plt.imshow(rgb_0_)
            plt.subplot(143)
            plt.imshow(rgb_0_ * weight_0[:,:,np.newaxis])
            plt.subplot(144)
            plt.imshow(rgb_weighted_0)
            plt.suptitle('Action '+ str(label[0].cpu().numpy()))
            plt.show()
            while True:
                try:
                    time.sleep(1)  # do something here
                    print('...')
                except KeyboardInterrupt:
                    print('\nPausing...  (Hit ENTER to continue, type quit to exit.)')
                    try:
                        response = raw_input()
                        if response == 'quit':
                            exit()
                        print('Resuming...')
                    except KeyboardInterrupt:
                        print('Resuming...')
                        continue
            # --------for debug end----------
            '''

            ls_cls = self.loss(output, label)
            result_frag.append(output.data.cpu().numpy())
            label_frag.append(label.data.cpu().numpy())
            loss = ls_cls

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['ls_cls'] = ls_cls.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['ls_cls'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['ls_cls']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, rgb, label in loader:
        #for rgb, label, weight in loader:

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)
            rgb = rgb.float().to(self.dev)
            #weight = weight.float().to(self.dev)
            #rgb = rgb * weight
            # inference
            with torch.no_grad():
                #output = self.model(rgb)
                output = self.model(data, rgb)

            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                ls_cls = self.loss(output, label)
                loss = ls_cls
                self.iter_info['ls_cls'] = ls_cls.data.item()
                loss_value.append(ls_cls.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        #print(self.result.size())
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['ls_cls']= np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k, 'eval')

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--fix_teacher', type=str2bool, default=True, help='set the teacher in evaluation mode')
        parser.add_argument('--teacher_weights', default=None, help='the learned weights of the teacher network')
        # endregion yapf: enable

        return parser
