#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:23:37 2019

@author: bruce
"""
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

frame_path = '/media/bruce/2Tssd/data/ntu_rgb_frames/'
openpose_path = '/media/bruce/2T/data/openpose/'
save_path = '/media/bruce/2Tssd/data/ntu_rgb_frames_crop/fivefs/'
debug = False

def filename_construct(setup_id, camera_id, subject_id, duplicate_id, action_id):
    skeleton_file_name = ''
    if setup_id/10 >= 1:
        skeleton_file_name = skeleton_file_name +'S0' + str(setup_id)
    else:
        skeleton_file_name = skeleton_file_name + 'S00' +  str(setup_id)

    if camera_id/10 >= 1:
        skeleton_file_name = skeleton_file_name + 'C0' +  str(camera_id)
    else:
        skeleton_file_name = skeleton_file_name + 'C00' +  str(camera_id)

    if subject_id/10 >= 1:
        skeleton_file_name = skeleton_file_name + 'P0' +  str(subject_id)
    else:
        skeleton_file_name = skeleton_file_name + 'P00' +  str(subject_id)

    if duplicate_id/10 >= 1:
        skeleton_file_name = skeleton_file_name + 'R0' +  str(duplicate_id)
    else:
        skeleton_file_name = skeleton_file_name + 'R00' +  str(duplicate_id)

    if action_id/10 >= 1:
        skeleton_file_name = skeleton_file_name + 'A0' +  str(action_id)
    else:
        skeleton_file_name = skeleton_file_name + 'A00' +  str(action_id)

    return skeleton_file_name

def openposeFile(frame_file, frame, skeleton_file_name, openpose_path):
    frame_file_ = frame_file + '/' + str(frame) + '.jpg'
    frame_ = '';
    if frame/100 >= 1:
        frame_ = str(frame)
    elif frame/10 >= 1:
        frame_ = '0' + str(frame)
    else:
        frame_ ='00' + str(frame)
    openpose_file_ = openpose_path + skeleton_file_name + '/' + skeleton_file_name + '_rgb_000000000'+ frame_ + '_keypoints.json'

    return openpose_file_, frame_file_

def cropBody(openpose_file, frame_file, action_id, flip):
    #upper=Image.new( 'RGB' , (224,112) , (0,0,0) )
    #lower=Image.new( 'RGB' , (224,112) , (0,0,0) )
    #whole=Image.new( 'RGB' , (224,448) , (0,0,0) )

    frame = Image.open(frame_file)

    if openpose_file:
        with open(openpose_file, 'r') as f:
            skeleton = json.load(f)

    # calculate which people?
    if len(skeleton['people']) == 1 or action_id < 50: # or action_id > 49:
        if len(skeleton['people']) < 1:
            return ''
        head_x = skeleton['people'][0]['pose_keypoints_2d'][0]
        head_y = skeleton['people'][0]['pose_keypoints_2d'][1]
        L_hand_x = skeleton['people'][0]['pose_keypoints_2d'][12]
        L_hand_y = skeleton['people'][0]['pose_keypoints_2d'][13]
        R_hand_x = skeleton['people'][0]['pose_keypoints_2d'][21]
        R_hand_y = skeleton['people'][0]['pose_keypoints_2d'][22]
        L_leg_x = skeleton['people'][0]['pose_keypoints_2d'][30]
        L_leg_y = skeleton['people'][0]['pose_keypoints_2d'][31]
        R_leg_x = skeleton['people'][0]['pose_keypoints_2d'][39]
        R_leg_y = skeleton['people'][0]['pose_keypoints_2d'][40]

        head = frame.crop((head_x-48, head_y - 48, head_x + 48, head_y + 48)) #2*3+1
        L_hand = frame.crop((L_hand_x-48, L_hand_y - 48, L_hand_x + 48, L_hand_y + 48))
        R_hand = frame.crop((R_hand_x-48, R_hand_y - 48, R_hand_x + 48, R_hand_y + 48))
        L_leg = frame.crop((L_leg_x-48, L_leg_y - 48, L_leg_x + 48, L_leg_y + 48))
        R_leg = frame.crop((R_leg_x-48, R_leg_y - 48, R_leg_x + 48, R_leg_y + 48))

        frame_concat=Image.new( 'RGB' , (96,480) , (0,0,0) )
        if flip:
            frame_concat.paste(head, (0,0))
            frame_concat.paste(L_hand, (0,96))
            frame_concat.paste(R_hand, (0,192))
            frame_concat.paste(L_leg, (0,288))
            frame_concat.paste(R_leg, (0,384))
            frame_concat=frame_concat.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            frame_concat.paste(head, (0,0))
            frame_concat.paste(L_hand, (0,96))
            frame_concat.paste(R_hand, (0,192))
            frame_concat.paste(L_leg, (0,288))
            frame_concat.paste(R_leg, (0,384))
        #print('frame_concat   if')
        return frame_concat

    elif len(skeleton['people']) > 1:
        head_x = skeleton['people'][0]['pose_keypoints_2d'][0]
        head_y = skeleton['people'][0]['pose_keypoints_2d'][1]
        L_hand_x = skeleton['people'][0]['pose_keypoints_2d'][12]
        L_hand_y = skeleton['people'][0]['pose_keypoints_2d'][13]
        R_hand_x = skeleton['people'][0]['pose_keypoints_2d'][21]
        R_hand_y = skeleton['people'][0]['pose_keypoints_2d'][22]
        L_leg_x = skeleton['people'][0]['pose_keypoints_2d'][30]
        L_leg_y = skeleton['people'][0]['pose_keypoints_2d'][31]
        R_leg_x = skeleton['people'][0]['pose_keypoints_2d'][39]
        R_leg_y = skeleton['people'][0]['pose_keypoints_2d'][40]

        head = frame.crop((head_x-24, head_y - 48, head_x + 24, head_y + 48)) #2*3+1
        L_hand = frame.crop((L_hand_x-24, L_hand_y - 48, L_hand_x + 24, L_hand_y + 48))
        R_hand = frame.crop((R_hand_x-24, R_hand_y - 48, R_hand_x + 24, R_hand_y + 48))
        L_leg = frame.crop((L_leg_x-24, L_leg_y - 48, L_leg_x + 24, L_leg_y + 48))
        R_leg = frame.crop((R_leg_x-24, R_leg_y - 48, R_leg_x + 24, R_leg_y + 48))

        head_x_1 = skeleton['people'][1]['pose_keypoints_2d'][0]
        head_y_1 = skeleton['people'][1]['pose_keypoints_2d'][1]
        L_hand_x_1 = skeleton['people'][1]['pose_keypoints_2d'][12]
        L_hand_y_1 = skeleton['people'][1]['pose_keypoints_2d'][13]
        R_hand_x_1 = skeleton['people'][1]['pose_keypoints_2d'][21]
        R_hand_y_1 = skeleton['people'][1]['pose_keypoints_2d'][22]
        L_leg_x_1 = skeleton['people'][1]['pose_keypoints_2d'][30]
        L_leg_y_1 = skeleton['people'][1]['pose_keypoints_2d'][31]
        R_leg_x_1 = skeleton['people'][1]['pose_keypoints_2d'][39]
        R_leg_y_1 = skeleton['people'][1]['pose_keypoints_2d'][40]

        head_1 = frame.crop((head_x_1-24, head_y_1 - 48, head_x_1 + 24, head_y_1 + 48)) #2*3+1
        L_hand_1 = frame.crop((L_hand_x_1-24, L_hand_y_1 - 48, L_hand_x_1 + 24, L_hand_y_1 + 48))
        R_hand_1 = frame.crop((R_hand_x_1-24, R_hand_y_1 - 48, R_hand_x_1 + 24, R_hand_y_1 + 48))
        L_leg_1 = frame.crop((L_leg_x_1-24, L_leg_y_1 - 48, L_leg_x_1 + 24, L_leg_y_1 + 48))
        R_leg_1 = frame.crop((R_leg_x_1-24, R_leg_y_1 - 48, R_leg_x_1 + 24, R_leg_y_1 + 48))

        frame_concat=Image.new( 'RGB' , (96,480) , (0,0,0) )
        if flip:
            frame_concat.paste(head, (0,0))
            frame_concat.paste(head_1, (48,0))
            frame_concat.paste(R_hand, (0,96))
            frame_concat.paste(R_hand_1, (48,96))
            frame_concat.paste(L_hand, (0,192))
            frame_concat.paste(L_hand_1, (48,192))
            frame_concat.paste(R_leg, (0,288))
            frame_concat.paste(R_leg_1, (48,288))
            frame_concat.paste(L_leg, (0,384))
            frame_concat.paste(L_leg_1, (48,384))
            frame_concat=frame_concat.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            frame_concat.paste(head, (0,0))
            frame_concat.paste(head_1, (48,0))
            frame_concat.paste(L_hand, (0,96))
            frame_concat.paste(L_hand_1, (48,96))
            frame_concat.paste(R_hand, (0,192))
            frame_concat.paste(R_hand_1, (48,192))
            frame_concat.paste(L_leg, (0,288))
            frame_concat.paste(L_leg_1, (48,288))
            frame_concat.paste(R_leg, (0,384))
            frame_concat.paste(R_leg_1, (48,384))
        #print('frame_concat   elif')
        return frame_concat
    else:
        #print(len(skeleton['people']))
        return ''

# to set the continue point S006C002P019R002A039



done = False
'''
for setup_id in range(1,21):     # 1:20 Diferernt height and distance
    if setup_id < sss:
        continue
    for camera_id in range(1,4):     # 1:3 camera views
        if setup_id < sss + 1 and camera_id < ccc:
            continue
        for subject_id in range(1,41):   # 1:40 distinct subjects aged between 10 to 35
            if setup_id < sss + 1 and camera_id < ccc + 1 and subject_id < ppp:
                continue
            for duplicate_id in range(1,3):  # 1:2 Performance action twice, one to left camera, one to right camera
                if setup_id < sss + 1 and camera_id < ccc + 1 and subject_id < ppp + 1 and duplicate_id < rrr:
                    continue
                for action_id in range(1,61):    # 1:60 Action class [11,12,30,31,53]
                    if setup_id < sss + 1 and camera_id < ccc + 1 and subject_id < ppp + 1 and duplicate_id < rrr +1 and action_id < aaa:
                        continue
'''
def construct_st_roi(filename, evaluation=False, random_interval=False,random_roi_move=False,random_flip=False):
    sequence_length = 6
    setup_id = int(
        filename[filename.find('S') + 1:filename.find('S') + 4])
    camera_id = int(
        filename[filename.find('C') + 1:filename.find('C') + 4])
    subject_id = int(
        filename[filename.find('P') + 1:filename.find('P') + 4])
    duplicate_id = int(
        filename[filename.find('R') + 1:filename.find('R') + 4])
    action_id = int(
        filename[filename.find('A') + 1:filename.find('A') + 4])

    skeleton_file_name = filename_construct(setup_id, camera_id, subject_id, duplicate_id, action_id)
    frame_file = frame_path + skeleton_file_name

    if os.path.isdir(frame_file):# and action_id == 50:

        # load the frames' file name from folder
        frames = os.listdir(frame_file)

        start_i = 0
        # checked all len(frames) are  > 6
        sample_interval = len(frames) // sequence_length
        flip = False
        if not evaluation:
            # Randomly choose sample interval and start frame
            start_i=0
            if random_interval:
                #print('random_interval:::::::::::::',random_interval)
                sample_interval = np.random.randint(1, len(frames) // sequence_length + 1)
                start_i = np.random.randint(0, len(frames) - sample_interval * sequence_length + 1)
            #if random_roi_move:

            if random_flip:
                flip = np.random.random() < 0.5

            # aline selection to the two sides

            frame_range = range(start_i, len(frames), sample_interval)
            #print(flip)
            #print(start_i, sample_interval)
        else:
            # Start at first frame and sample uniformly over sequence
            start_i = 0
            flip = False
            frame_range = range(start_i, len(frames), sample_interval)

        fivefs_concat=Image.new( 'RGB' , (480,480) , (0,0,0) )
        i=0
        for frame in frame_range:

            if frame != 0 and frame != (6*sample_interval):

                #print(frame)
                if not debug:
                    #openpose_file_, frame_file_ = openposeFile(frame_file, frame, skeleton_file_name, openpose_path)
                    frame_croped = ''
                    frame_ = frame
                    # find the closest non'' frame
                    while frame_croped == '':
                        openpose_file_, frame_file_ = openposeFile(frame_file, frame_, skeleton_file_name, openpose_path)
                        # both openpose and RGB frame should exist
                        if os.path.isfile(openpose_file_) and os.path.isfile(frame_file_):
                            frame_croped = cropBody(openpose_file_, frame_file_, action_id, flip)
                            #print('file consistent: ',openpose_file_)
                        else:
                            #print('file_unconsistent: ',openpose_file_, os.path.isfile(openpose_file_))
                            string = str(frame_file_ + " {}\n" + openpose_file_ + ' {}\n').format(os.path.isfile(frame_file_), os.path.isfile(openpose_file_))
                            with open('file_unconsistent_crop.txt', 'a') as fd:
                                fd.write(f'\n{string}')

                        frame_ = frame_ + 1
                        if frame_ > len(frames):
                            frame_croped = Image.new( 'RGB' , (96,480) , (0,0,0) )
                            break

                    fivefs_concat.paste(frame_croped, (i*96+1,0))
                    i+=1
            '''
            plt.imshow(fivefs_concat)
            plt.suptitle('Corpped Body Parts')
            plt.show()
            '''
        #frames_save = save_path + skeleton_file_name +'.png'
        #fivefs_concat.save(frames_save,"PNG")
    return fivefs_concat

if __name__ == '__main__':
    '''
    import os
    import json
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from PIL import Image
    import numpy as np

    frame_path = '../../data/NWestern_UCLA/multiview_action/'
    openpose_path = '../../data/NWestern_UCLA/openpose_ucla/'
    save_path = '../../data/NWestern_UCLA/ucla_rgb_frames_crop/fivefs_240/'
    debug = False
    duplicate_list=[]
    frame_croped = transform(frame_croped)
    frame_croped = torch.flip(frame_croped, (-1,))
    frame_croped = np.einsum('kli->lik',frame_croped.numpy())
    '''

    fivefs_concat = construct_st_roi('S001C001P001R001A054', evaluation=False,random_interval=True, random_flip=True)
    #fivefs_concat = np.einsum('kli->lik',fivefs_concat.numpy())
    plt.imshow(fivefs_concat)
    plt.suptitle('Corpped Body Parts')
    plt.show()
