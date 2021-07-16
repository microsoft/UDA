# MetaAlign-Detection

This is the repo for reproducing the results for UDA of object detection task in the paper 'MetaAlign: Coordinating Domain Alignment and Classification for Unsupervised Domain Adaptation'. We take [DA_Detection](https://github.com/VisionLearningGroup/DA_Detection) as our code framework.

## Prerequisites

- pytorch==0.4.0
- torchvision==0.2.1
- python==3.6
- cuda==9.0

## Dataset

- __PASCAL_VOC 07+12__: Please folow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datsets.
- __WaterColor__: Please follow the dataset preparation insturction in [Cross Domain Detection](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets). 

## Run

train
```bash
python trainval_net_global_local_ml.py --dataset pascal_voc_water --dataset_t water --net res101 --cuda --lc --gc --meta_lr 10.0 --lr 0.001 --save_dir exp --bs 1
```

test
```bash
python test_net_global_local.py --lc --gc --dataset  clipart --net res101 --cuda --load_name path_to_ckpt_file 
```
[pre-trained models](https://drive.google.com/file/d/17XG4-lmvEwCYNi_UTgLAE6ASlP3cnXYg/view?usp=sharing)

## Citation

```
@inproceedings{wei2021metaalign,
  title={MetaAlign: Coordinating Domain Alignment and Classification for Unsupervised Domain Adaptation},
  author={Wei, Guoqiang and Lan, Cuiling and Zeng, Wenjun and Chen, Zhibo},
  booktitle={CVPR},
  pages={16643--16653},
  year={2021}
}

```
