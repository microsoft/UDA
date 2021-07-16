# MetaAlign-Classification

This is the repo for reproducing the results for UDA of classification task in the paper 'MetaAlign: Coordinating Domain Alignment and Classification for Unsupervised Domain Adaptation'. We take [GVB](https://github.com/cuishuhao/GVB) as our code framework.

## Prerequisites

- pytorch==1.0.1
- python==3.6
- cuda==10.0

## Dataset

Please download the office-home dataset from the [official website](https://www.hemanthdv.org/officeHomeDataset.html).

## Run

```bash
python train.py --gvb --seed 1 --meta_lr 100.0 --s_dset_path data/office-home/Art.txt --t_dset_path data/office-home/Clipart.txt --output_root exp_local --gvbg  --output_dir art2clipart_0 --dset office-home
```

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
