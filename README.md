# Undoing the Damage of Label Shift for Cross-domain Semantic Segmentation (CVPR 2022)
This is a [pytorch](http://pytorch.org/) implementation of Undoing the Damage of Label Shift for Cross-domain Semantic Segmentation.

###  Environment Requirements
- Python 3.6
- Pytorch 1.2.0
- torchvision from master
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 9.0
### Step-by-step installation

```bash
conda create --name undoing -y python=3.6
conda activate undoing

# this installs the right pip and dependencies for the fresh python
conda install -y ipython pip

pip install ninja yacs cython matplotlib tqdm opencv-python imageio mmcv

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.2
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
```

### Getting started

#### Dataset
- Download [The GTA5 Dataset]( https://download.visinf.tu-darmstadt.de/data/from_games/ )

- Download [The SYNTHIA Dataset]( http://synthia-dataset.net/download/808/ )

- Download [The Cityscapes Dataset]( https://www.cityscapes-dataset.com/ )

- Symlink the required dataset
```bash
ln -s /path_to_gta5_dataset datasets/gta5
ln -s /path_to_synthia_dataset datasets/synthia
ln -s /path_to_cityscapes_dataset datasets/cityscapes
```

- Generate the label statics file for GTA5 and SYNTHIA Datasets by running 
```
python datasets/generate_gta5_label_info.py -d datasets/gta5 -o datasets/gta5/
python datasets/generate_synthia_label_info.py -d datasets/synthia -o datasets/synthia/
```

The data folder should be structured as follows:
```
├── datasets/
│   ├── cityscapes/     
|   |   ├── gtFine/
|   |   ├── leftImg8bit/
│   ├── gta5/
|   |   ├── images/
|   |   ├── labels/
|   |   ├── gtav_label_info.p
│   ├── synthia/
|   |   ├── RAND_CITYSCAPES/
|   |   ├── synthia_label_info.p
│   └── 			
...
```
### Pretrained models

- For original ASPP, please download the pretrained models of FADA for [Synthia -> CityScapes task(s2c) and GTA5 -> CityScapes task(g2c)](https://drive.google.com/drive/folders/1M7mwfSX3fx4W9KUevZwdmCo4JISnfCI_).

- For modified ASPP, please reproduce [FADA](https://github.com/JDAI-CV/FADA) by changing the [classifer](https://github.com/JDAI-CV/FADA/blob/master/core/models/classifier.py). And we also provide the pretrainded models [here](https://drive.google.com/drive/folders/1wQF_bKSij7XpNvADcFMorIsi9dxiJcv4?usp=sharing).

### Train (We provide the training script using 4 GPUs and original ASPP）

#### Label Distribution Estimation for target
```
python test_trainlabelfre.py -cfg configs/deeplabv2_r101_adv.yaml resume g2c_adv.pth
```
#### Inference Adjustment (IA)
- Improve the adversial result of FADA by IA
```
python test_IA.py -cfg configs/deeplabv2_r101_tgt_self_distill.yaml resume g2c_adv.pth
```
- Improve the pseudo label by IA
```
python test_IA.py -cfg configs/deeplabv2_r101_adv.yaml --saveres resume g2c_adv.pth OUTPUT_DIR datasets/cityscapes/soft_labels DATASETS.TEST cityscapes_train

python -m torch.distributed.launch --nproc_per_node=4 train_self_distill.py -cfg configs/deeplabv2_r101_tgt_self_distill_2.yaml OUTPUT_DIR results/sd_test
```
#### Classifier Refinement (CR)
- Improve the adversial result of FADA by CR
```
python -m torch.distributed.launch --nproc_per_node=4 train_CR.py -cfg configs/deeplabv2_r101_adv.yaml OUTPUT_DIR results/adv_test_CR resume g2c_adv.pth

python test.py -cfg configs/deeplabv2_r101_tgt_self_distill.yaml resume results/adv_test_CR
```
- Improve the pseudo label by CR
```
python -m torch.distributed.launch --nproc_per_node=4 train_CR.py -cfg configs/deeplabv2_r101_adv.yaml OUTPUT_DIR results/adv_test_CR resume g2c_adv.pth

python test.py -cfg configs/deeplabv2_r101_adv.yaml --saveres resume results/adv_test_CR/xxx.pth OUTPUT_DIR datasets/cityscapes/soft_labels DATASETS.TEST cityscapes_train

python -m torch.distributed.launch --nproc_per_node=4 train_self_distill.py -cfg configs/deeplabv2_r101_tgt_self_distill_2.yaml OUTPUT_DIR results/sd_test
```
#### Connect with other self-training methods
- Please refer to the code of [IAST](https://github.com/Raykoooo/IAST) and [ProDA](https://github.com/microsoft/ProDA). 
- PS: for ProDA, you should change the model structure and the data preprocessing as FADA.
- Please contact lyhaolive@gmail.com for more details.


### Evaluate
```
python test.py -cfg configs/deeplabv2_r101_tgt_self_distill.yaml resume g2c_adv.pth

python test.py -cfg configs/deeplabv2_r101_tgt_self_distill_2.yaml resume g2c_sd.pth
```

### Acknowledge
Codes are adapted from [FADA](https://github.com/JDAI-CV/FADA), [IAST](https://github.com/Raykoooo/IAST) and [ProDA](https://github.com/microsoft/ProDA). We thank them for their excellent projects.

### Citation
If you find this code useful please consider citing
```

@InProceedings{Liu_2022_CVPR
author = {Liu, Yahao and Deng, Jinhong and Tao, Jiale and Chu, Tong and Duan, Lixin and Li, Wen},
title = {Undoing the Damage of Label Shift for Cross-domain Semantic Segmentation },
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition(CVPR)},
year = {2022}
}
```
