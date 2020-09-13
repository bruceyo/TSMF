# Teacher-Student-Network
Teacher-Student Network: a Model-Based Multimodal Fusion Method for Action Recognition

## Introduction
<!--
This repository holds the codebase, dataset and models for the paper:
**Teacher-Student Network: a Model-Based Multimodal Fusion Method for Action Recognition**

update github version with below commands:
  git add .
  git add commit
  git push git@github.com:bruceyo/Teacher-Student-Network.git
-->
<div align="center">
    <img src="resource/info/neural_fused_repre.png">
</div>

## Prerequisites
- Python3 (>3.5)
- [PyTorch](http://pytorch.org/)
- [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) **with** [Python API](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md#python-api). (Optional: for training purpose)
- Other Python libraries can be installed by `pip install -r requirements.txt`

## Installation
``` shell
cd torchlight; python setup.py install; cd ..
```

## Get pretrained models

Download pretrained models from [GoogleDrive](https://drive.google.com/drive/folders/10lE7HcHbr60Arm1XlQxW5Mz7UQb_QM0l?usp=sharing), and manually put them into ```./trained_models```.

## Data Preparation
### Datasets
#### NTU RGB+D
NTU RGB+D can be downloaded from [their website](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp).
The **3D skeletons**(5.8GB) modality and the RGB modality are required in our experiments. After that, this command should be used to build the database for training or evaluation:
```
python tools/ntu_gendata.py --data_path <path to nturgbd+d_skeletons>
```
where the ```<path to nturgbd+d_skeletons>``` points to the 3D skeletons modality of NTU RGB+D dataset you download.

For evaluation, the processed data invludes: ```val_data and val_label``` are available from [GoogleDrive](https://drive.google.com/drive/folders/1D7zXKuk4YF4vGczrkMv87lapdMlwEy_S?usp=sharing). Please manually put it in folder: ```./data/NTU_RGBD```

#### Northwestern-UCLA Multiview
The Multiview 3D event dataset is capture by [Wangjian](http://wangjiangb.github.io/my_data.html) and Xiaohan Nie in UCLA. It contains RGB, depth and human skeleton data captured simultaneously by three Kinect cameras. This dataset include 10 action categories: pick up with one hand, pick up with two hands, drop trash, walk around, sit down, stand up, donning, doffing, throw, carry. Each action is performed by 10 actors. This dataset contains data taken from a variety of viewpoints.

The dataset can be found in [part-1](http://users.eecs.northwestern.edu/~jwa368/data/multiview_action.tgz.part-aa), [part-2](http://users.eecs.northwestern.edu/~jwa368/data/multiview_action.tgz.part-ab), [part-3](http://users.eecs.northwestern.edu/~jwa368/data/multiview_action.tgz.part-ac), [part-4](http://users.eecs.northwestern.edu/~jwa368/data/multiview_action.tgz.part-ad), [part-5](http://users.eecs.northwestern.edu/~jwa368/data/multiview_action.tgz.part-ae), [part-6](http://users.eecs.northwestern.edu/~jwa368/data/multiview_action.tgz.part-af), [part-7](http://users.eecs.northwestern.edu/~jwa368/data/multiview_action.tgz.part-ag), [part-8](http://users.eecs.northwestern.edu/~jwa368/data/multiview_action.tgz.part-ah), [part-9](http://users.eecs.northwestern.edu/~jwa368/data/multiview_action.tgz.part-ai), [part-10](http://users.eecs.northwestern.edu/~jwa368/data/multiview_action.tgz.part-aj), [part-11](http://users.eecs.northwestern.edu/~jwa368/data/multiview_action.tgz.part-ak), [part-12](http://users.eecs.northwestern.edu/~jwa368/data/multiview_action.tgz.part-al), [part-13](http://users.eecs.northwestern.edu/~jwa368/data/multiview_action.tgz.part-am), [part-14](http://users.eecs.northwestern.edu/~jwa368/data/multiview_action.tgz.part-an), [part-15](http://users.eecs.northwestern.edu/~jwa368/data/multiview_action.tgz.part-ao), [part-16](http://users.eecs.northwestern.edu/~jwa368/data/multiview_action.tgz.part-ap).

RGB videos could be downloaded from:  [RGB videos](http://users.eecs.northwestern.edu/~jwa368/data/multiview_action_videos.tgz), which is used to generate 2D skeleton data by using OpenPose.

For evaluation and training, the processed dataset is available from [GoogleDrive](https://drive.google.com/drive/folders/1qxjhtuxuTVIdMMxexe0szDImedkT8v94?usp=sharing). Please manually put it in folder: ```./data/NW-UCLA```

### 2D Skeleton Retrieval from the RGB Video Input
After installed the Openpose tool, run
```
su
sh tools/2D_Retrieve_<dataset>.sh
```
where the ```<dataset>``` must be ```ntu_rgbd``` or ```ucla```, depending on the dataset you want to use.

### Generate Region of Interest
```
python tools/gen_fivefs_<dataset>
```
where the ```<dataset>``` must be ```ntu_rgbd``` or ```ucla```, depending on the dataset you want to use.

The processed ROI of NTU-RGB+D is available from [GoogleDrive](https://drive.google.com/file/d/1NjLSNaJjR-XuSv3MmrQisFJTTg-Vc8ID/view?usp=sharing);
The processed ROI of NW_UCLA is available from [GoogleDrive](https://drive.google.com/file/d/1A5QdFCG4qLAxV4g7aRXgsy-uWOd1bD8W/view?usp=sharing).
## Testing Pretrained Models
<!-- ### Evaluation
Once datasets and the pretrained models are ready, we can start the evaluation. -->
### Evaluate on NTU-RGB+D
For **cross-subject** evaluation in **NTU RGB+D**, run
```
python main_student.py recognition -c config/ntu_rgbd/xsub/student_test.yaml
```
Check the emsemble:
```
python ensemble_ntu_rgbd.py --datasetsNTU_RGBD/xsub
```
For **cross-view** evaluation in **NTU RGB+D**, run
```
python main_student.py recognition -c config/ntu_rgbd/xview/student_test.yaml
```
Check the ensemble:
```
python ensemble_ntu_rgbd.py --datasets NTU_RGBD/xview
```
### Evaluate on NW_UCLA multiview_action
```
python main_student.py recognition -c config/ucla/student_test.yaml
```
Check the emsemble:
```
python ensemble_nw_ucla.py
```

## Training
To train a new Teacher-Student model, run
```
python main_student.py recognition -c config/<dataset>/train_student.yaml [--work_dir <work folder>]
```
where the ```<dataset>``` must be ```ntu_rgbd/xsub```, ```ntu_rgbd/xview``` or ```ucla```, depending on the dataset you want to use.
The training results, including **model weights**, configurations and logging files, will be saved under the ```./work_dir``` by default or ```<work folder>``` if you appoint it.

You can modify the training parameters such as ```work_dir```, ```batch_size```, ```step```, ```base_lr``` and ```device``` in the command line or configuration files. The order of priority is:  command line > config file > default parameter. For more information, use ```main.py -h```.

## Evaluation
Finally, custom model evaluation can be achieved by this command as we mentioned above:
```
python main_student.py recognition -c config/<dataset>/student_test.yaml --weights <path to model weights>
```

## Contact
For any question, feel free to contact
```
xxx    : xxx@xxx
xxx    : xxx@gmail.com
```
