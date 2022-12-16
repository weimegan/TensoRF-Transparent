# RETrO: Rendering and Extracting Transparent Objects using TensoRF and Feature Field Distillation
## [Project page](https://megan-kate-anushka.github.io/) 
This repository extends the work of [Decomposing NeRF for Editing via Feature Field Distillation](https://arxiv.org/abs/2205.15585) to render and extract features from 3D scenes that include transparent objects. We use
[TensoRF: Tensorial Radiance Fields](https://arxiv.org/abs/2203.09517) instead of NeRF for faster scene rendering. We use feature field distillation to render specific features within a scene. 

## Installation

#### Tested on Ubuntu 20.04 + Pytorch 1.10.1 

Install environment:
```
conda create -n RETrO python=3.8
conda activate RETrO
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
```


## Datasets
* [Synthetic-NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 
* [KeyPose](https://sites.google.com/view/keypose/home)
* [Dex-NeRF](https://sites.google.com/view/dex-nerf)



## Quick Start
The training script is in `train.py`, to train RETrO:

```
python train.py --config configs/flower.txt
```


we provide a few examples in the configuration folder, please note:

 `dataset_name`, choices = ['llff', 'dexnerfrealtable', 'llff_features'];

 `shadingMode`, choices = ['MLP_Fea', 'SH'];

 `model_name`, choices = ['TensorVMSplit', 'TensorCP'], corresponding to the VM and CP decomposition. 
 You need to uncomment the last a few rows of the configuration file if you want to training with the TensorCP model；

 `n_lamb_sigma` and `n_lamb_sh` are string type refer to the basis number of density and appearance along XYZ
dimension;

 `N_voxel_init` and `N_voxel_final` control the resolution of matrix and vector;

 `N_vis` and `vis_every` control the visualization during training;

  You need to set `--render_test 1`/`--render_path 1` if you want to render testing views or path after training. 

More options refer to the `opt.py`. 

### For pretrained checkpoints and results please see:
[https://1drv.ms/u/s!Ard0t_p4QWIMgQ2qSEAs7MUk8hVw?e=dc6hBm](https://1drv.ms/u/s!Ard0t_p4QWIMgQ2qSEAs7MUk8hVw?e=dc6hBm)



## Rendering

```
python train.py --config configs/flower.txt --ckpt checkpoints/flower.th --feat_ckpt log/tensorf_flower_VM_features/tensorf_flower_VM_features.th --query flower --render_only 1 --render_test 1 
```

You can just simply pass `--render_only 1` and `--ckpt path/to/your/checkpoint` to render images from a pre-trained
checkpoint. You may also need to specify what you want to render, like `--render_test 1`, `--render_train 1` or `--render_path 1`.
The rendering results are located in your checkpoint folder. 

## Extracting mesh
You can also export the mesh by passing `--export_mesh 1`:
```
python train.py --config configs/flower.txt --ckpt path/to/your/checkpoint --export_mesh 1
```
Note: Please re-train the model and don't use the pretrained checkpoints provided by us for mesh extraction, 
because some render parameters has changed.

## Training with your own data
We provide two options for training on your own image set:

1. Following the instructions in the [NSVF repo](https://github.com/facebookresearch/NSVF#prepare-your-own-dataset), then set the dataset_name to 'tankstemple'.
2. Calibrating images with the script from [NGP](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md):
`python dataLoader/colmap2nerf.py --colmap_matcher exhaustive --run_colmap`, then adjust the datadir in `configs/your_own_data.txt`. Please check the `scene_bbox` and `near_far` if you get abnormal results.
    

## Citation
If you find our code helpful, please consider citing:
```
@misc{retro,
  title={RETrO: Rendering and Extracting Transparent Objects using TensoRF and Feature Field Distillation},
  author={Wei, Megan and Xu, Katherine and Ray, Anushka},
  journal={Github repository},
  year={2022}
}
```
