import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
# python ensemble.py` --datasets ntu/xview
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='NTU_RGBD/xsub', choices={'NTU_RGBD/xsub', 'NTU_RGBD/xview'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
arg = parser.parse_args()

dataset = arg.datasets
#work_dir = './work_dir/ucla/skeleton_weight_rgb/ensemble'
label = open('./data/'+dataset+'/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open('./work_dir/'+dataset.lower()+'/student/test_result.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('./work_dir/'+dataset.lower()+'/teacher/test_result.pkl', 'rb')
r2 = list(pickle.load(r2).items())
confusion_matrix = np.zeros([len(label[0]), len(label[0])])
right_num = total_num = right_num_5 = 0
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    r = r11 + r22
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
    confusion_matrix[int(l)][r] += 1

acc = right_num / total_num
acc5 = right_num_5 / total_num
print(acc, acc5)
#np.savetxt(os.path.join(work_dir,'confusion_matrix_epoch_ensemble.csv'), confusion_matrix, fmt='%d', delimiter=",")
