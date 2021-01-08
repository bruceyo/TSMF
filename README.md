# TSMF Model
This repository holds the codebase, dataset and models for the paper:

**Multimodal Fusion via Teacher-Student Network for Indoor Action Recognition** Bruce X.B. Yu, Yan Liu, Keith C.C. , AAAI 2021

<div align="center">
    <img src="resource/info/neural_fused_repre.png">
</div>

## Abstract
<!--
This repository holds the codebase, dataset and models for the paper:
**Multimodal Fusion via Teacher-Student Network for Indoor Action Recognition**

update github version with below commands:
  git add .
  git add commit
  git push git@github.com:bruceyo/TSMF.git
-->

Indoor action recognition plays an important role in modern society, such as intelligent healthcare in large mobile cabin hospital. With the wide usage of depth sensors like Kinect, multimodal information including skeleton and RGB modalities brings a promising way to improve the performance. However, existing methods are either focusing on single data modality or failed to take the advantage of multiple data modalities. In this paper, we propose a Teacher-Student Multimodal Fusion (TSMF) model that fuses the skeleton and RGB modalities at model level for indoor action recognition. In our TSMF, we utilize a teacher network to transfer the structural knowledge of the skeleton modality to a student network for the RGB modality. With extensively experiments on two benchmarking datasets: NTU RGB+D and PKU-MMD, results show that the proposed TSMF consistently performs better than state-of-the-art single modal and multimodal methods. It also indicates that our TSMF could not only improve the accuracy of the student network but also significantly improve the ensemble accuracy.

## Prerequisites
- Python3 (>3.5)
- [PyTorch](http://pytorch.org/)
- [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) **with** [Python API](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md#python-api). (Optional: for evaluation purpose)
- Other Python libraries can be installed by `pip install -r requirements.txt`

## Installation
``` shell
cd torchlight; python setup.py install; cd ..
```

## Get pretrained models

Download pretrained models from [GoogleDrive](https://drive.google.com/drive/folders/1J63NA-L8v6FofNip4MMt3Zfi_S-FALt2?usp=sharing), and manually put them into ```./trained_models```.

## Data Preparation
### Datasets
#### NTU RGB+D
NTU RGB+D can be downloaded from [their website](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp).
The **3D skeletons**(5.8GB) modality and the RGB modality are required in our experiments. After that, this command should be used to build the database for training or evaluation:
```
python tools/ntu_gendata.py --data_path <path to nturgbd+d_skeletons>
```
where the ```<path to nturgbd+d_skeletons>``` points to the 3D skeletons modality of NTU RGB+D dataset you download.

For evaluation, the processed data includes: ```val_data and val_label``` are available from [GoogleDrive](https://drive.google.com/drive/folders/1D7zXKuk4YF4vGczrkMv87lapdMlwEy_S?usp=sharing). Please manually put it in folder: ```./data/NTU_RGBD```

#### PKU-MMD
The dataset can be found in [PKU-MMD](https://github.com/ECHO960/PKU-MMD). PKU-MMD is a large action recognition dataset that contains 1076 long video sequences in 51 action categories, performed by 66 subjects in three camera views. It contains almost 20,000 action instances and 5.4 million frames in total. We transfer the 3D skeleton modality to seperate action repetition files with the command:
```
python tools/utils/skeleton_to_ntu_format.py
```
After that, this command should be used to build the database for training or evaluation:
```
python tools/pku_gendata.py --data_path <path to pku_mmd_skeletons>
```
where the ```<path to nturgbd+d_skeletons>``` points to the 3D skeletons modality of PKU-MMD dataset you processed with the above command.

For evaluation, the processed data includes: ```val_data and val_label``` are available from [GoogleDrive](https://drive.google.com/drive/folders/1iwsf1RP0a8rWLoh55kHlFibB3edQtd01?usp=sharing). Please manually put it in folder: ```./data/PKU_MMD```

### 2D Skeleton Retrieval from the RGB Video Input
After installed the Openpose tool, run
```
su
sh tools/2D_Retrieve_<dataset>.sh
```
where the ```<dataset>``` must be ```ntu_rgbd``` or ```pku_mmd```, depending on the dataset you want to use.

### Generate Region of Interest
```
python tools/gen_fivefs_<dataset>
```
where the ```<dataset>``` must be ```ntu_rgbd``` or ```pku_mmd```, depending on the dataset you want to use.

The processed ROI of NTU-RGB+D is available from [GoogleDrive](https://drive.google.com/file/d/1NjLSNaJjR-XuSv3MmrQisFJTTg-Vc8ID/view?usp=sharing);
The processed ROI of PKU-MMD is available from [GoogleDrive](https://drive.google.com/file/d/1zHtjWF06mHjcMLsRhTIFiLPu9wpfoYs8/view?usp=sharing).
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
python ensemble.py --datasets ntu_xsub
```
For **cross-view** evaluation in **NTU RGB+D**, run
```
python main_student.py recognition -c config/ntu_rgbd/xview/student_test.yaml
```
Check the ensemble:
```
python ensemble.py --datasets ntu_xview
```
### Evaluate on PKU-MMD
For **cross-subject** evaluation in **PKU-MMD**, run
```
python main_student.py recognition -c config/pku_mmd/xsub/student_test.yaml
```
Check the emsemble:
```
python ensemble.py --datasets pku_xsub
```
For **cross-view** evaluation in **PKU-MMD**, run
```
python main_student.py recognition -c config/pku_mmd/xview/student_test.yaml
```
Check the emsemble:
```
python ensemble.py --datasets pku_xview
```

## Training
To train a new TSMF model, run
```
python main_student.py recognition -c config/<dataset>/train_student.yaml [--work_dir <work folder>]
```
where the ```<dataset>``` must be ```ntu_rgbd/xsub```, ```ntu_rgbd/xview```, ```pku_mmd/xsub``` or ```pku_mmd/xview```, depending on the dataset you want to use.
The training results, including **model weights**, configurations and logging files, will be saved under the ```./work_dir``` by default or ```<work folder>``` if you appoint it.

You can modify the training parameters such as ```work_dir```, ```batch_size```, ```step```, ```base_lr``` and ```device``` in the command line or configuration files. The order of priority is:  command line > config file > default parameter. For more information, use ```main_student.py -h```.

## Evaluation
Finally, custom model evaluation can be performed by this command as we mentioned above:
```
python main_student.py recognition -c config/<dataset>/student_test.yaml --weights <path to model weights>
```

## Acknowledgements
This repo is based on
- [ST-GCN](https://github.com/yysijie/st-gcn)
- [2s-AGCN](https://github.com/lshiwjx/2s-AGCN)
- [MS-G3D](https://github.com/kenziyuliu/MS-G3D)

Thanks to the original authors for their work!

## Citation
Please cite the following paper if you use this repository in your reseach.
```
@inproceedings{tsmf2021aaai,
  title     = {Multimodal Fusion via Teacher-Student Network for Indoor Action Recognition},
  author    = {Bruce X.B. Yu and Yan Liu and Keith C.C. Chan},
  booktitle = {AAAI},
  year      = {2021},
}
```

## Contact
For any question, feel free to contact
```
Bruce Yu: bruceyo2011@gmail.com
```
