import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
# python ensemble.py` --datasets ntu/xview
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='xsub', choices={'kinetics', 'ntu/xsub', 'ntu/xview'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
arg = parser.parse_args()

dataset = arg.datasets
label = open('/media/bruce/2Tssd/data/pku_st_gcn/xview/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
#/media/bruce/2T/data/st-gcn/skl-test/yansijie-xsub
#/media/bruce/2T/data/st-gcn/with-rgb-test/full_rgb_crop_sklweight_1
r1 = open('../../data/2s-agcn/ntu/' + dataset + '/agcn_test_joint/epoch1_test_score.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('../../data/2s-agcn/ntu/' + dataset + '/agcn_test_bone/epoch1_test_score.pkl', 'rb')
r2 = list(pickle.load(r2).items())
#r2 = open('../../data/st-gcn/skl-test/yansijie-xsub/test_result.pkl', 'rb')
#r2 = list(pickle.load(r2).items())
# 20200321 bruce
#r3 = open('/media/bruce/2T/data/st-gcn/with-rgb-test/full_rgb_crop_skl_pre_t_3_2ndfusion_2/test_result.pkl', 'rb')
#r3 = list(pickle.load(r3).items())
# 20200321 bruce
r3 = open('../../data/st-gcn-pku/rgb_cv/test_result_epoch80.pkl', 'rb') # cv: rgb_cs/test_result_epoch60.pkl'
r3 = list(pickle.load(r3).items())

r1_yan = open('../../data/st-gcn-pku/skl_cv_1/test_result_epoch60.pkl', 'rb')
r1_yan = list(pickle.load(r1_yan).items())
r2_yan = open('../../data/st-gcn-pku/bone_cv_1/test_result_epoch25.pkl', 'rb')
r2_yan = list(pickle.load(r2_yan).items())

right_num = total_num = right_num_5 = 0
right_num_r11 = right_num_r22 = right_num_r33 = right_num_r1r2 = right_num_r1r3 = right_num_stgcn = 0
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    _, r33 = r3[i]

    _, r11_yan = r1_yan[i]
    _, r22_yan = r2_yan[i]
    # r = r11 + r22 * arg.alpha  + r33 * arg.alpha
    r1r2 = r11_yan + r22_yan * arg.alpha
    r1r3 = r11_yan + r33 * arg.alpha
    r_stgcn = r11_yan + r22_yan * arg.alpha  + r33 * arg.alpha

    #rank_5 = r.argsort()[-5:]
    #right_num_5 += int(int(l) in rank_5)
    #r = np.argmax(r)
    r11_yan = np.argmax(r11_yan)
    r22_yan = np.argmax(r22_yan)
    r33 = np.argmax(r33)
    r1r2 = np.argmax(r1r2)
    r1r3 = np.argmax(r1r3)
    r_stgcn = np.argmax(r_stgcn)

    #right_num += int(r == int(l))
    right_num_r11 += int(r11_yan == int(l))
    right_num_r22 += int(r22_yan == int(l))
    right_num_r33 += int(r33 == int(l))
    right_num_r1r2 += int(r1r2 == int(l))
    right_num_r1r3 += int(r1r3 == int(l))
    right_num_stgcn += int(r_stgcn == int(l))

    total_num += 1
#acc = right_num / total_num
acc_r11 = right_num_r11 / total_num
acc_r22 = right_num_r22 / total_num
acc_r33 = right_num_r33 / total_num
acc_r1r2 = right_num_r1r2 / total_num
acc_r1r3 = right_num_r1r3 / total_num
acc_stgcn = right_num_stgcn / total_num
acc5 = right_num_5 / total_num
#print(acc, acc5)
print('ST-GCN Joint Top 1:         ',acc_r11)
print('ST-GCN Bone Top 1:          ',acc_r22)
print('ST-ROI Top 1:               ',acc_r33)
print('Joint + bone Top 1:         ',acc_r1r2)
print('Joint + ST-ROI Top 1:       ',acc_r1r3)
print('Joint + bone + ST-ROI Top 1:',acc_stgcn)
'''
result= r1_yan
class_num=51
instance_num = len(result)
#rank = result.argsort()
confusion_matrix = np.zeros([class_num, class_num])

for i in range(instance_num):
    _, true_l = label[:,i]
    _, pred_l = result[i]
    pred_l = np.argmax(pred_l)
    confusion_matrix[int(true_l)][pred_l] += 1
#np.savetxt("confusion_matrix.csv", confusion_matrix, fmt='%.3e', delimiter=",")
np.savetxt(os.path.join('../../data/st-gcn/models/yansijie','confusion_matrix_cs.csv'), confusion_matrix, fmt='%d', delimiter=",")
'''
