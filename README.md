# ScaLR

PyTorch code and models for VaViT.

[**Vanilla ViT for Automotive Point Cloud Semantic Segmentation**]()\
[*Gilles Puy*<sup>1</sup>](https://sites.google.com/site/puygilles/home),
[*Nermin Samet*<sup>1</sup>](https://nerminsamet.github.io/)
[*Alexandre Boulch*<sup>1</sup>](http://boulch.eu),
[*Spyros Gidaris*<sup>1</sup>](https://scholar.google.fr/citations?user=7atfg7EAAAAJ&hl=en),
[*Tuan-Hung VU*<sup>1</sup>](https://tuanhungvu.github.io/)
[*Renaud Marlet*<sup>1,2</sup>](http://imagine.enpc.fr/~marletr/)  

<sup>1</sup>*valeo.ai, France*\
<sup>2</sup>*LIGM, Ecole des Ponts, Univ Gustave Eiffel, CNRS, France*.


<p align="center">
  <img src=./illustration.png alt="VaViT architecture"/>
</p>

If you find this code or work useful, please cite the following [paper]():
```
@article{vavit,
  title={Vanilla ViT for Automotive Point Cloud Semantic Segmentation},
  author={Puy, Gilles and Samet, Nermin and Boulch, Alexandre and Gidaris, Spyros and Vu, Tuan-Hung and Marlet, Renaud},
  booktitle={arXiv},
  year={2026}
}
```

## Overview

- [Installation](#installation)
- [Datasets](#datasets)
- [Available models](#available-models)
- [Evaluation](#evaluation)
- [Training](#training)


## Installation

### Environment

The results were obtained with the environment below.
```bash
conda create -n vavit
conda activate vavit
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch
pip install pyaml==25.7.0 tqdm==4.67.1 scipy==1.15.3 tensorboard==2.20.0
git clone https://github.com/valeoai/VaViT
cd VaViT
```

Download and untar the following [file](https://github.com/valeoai/WaffleIron/files/10294733/info_datasets.tar.gz):
```bash
wget https://github.com/valeoai/WaffleIron/files/10294733/info_datasets.tar.gz
tar -xvzf info_datasets.tar.gz
rm info_datasets.tar.gz
```


### Datasets

We use the following datasets: [nuScenes](https://www.nuscenes.org/nuscenes), [SemanticKITTI](https://www.semantic-kitti.org/) and [Waymo Open Dataset](https://waymo.com/open/).

Please download nuScenes and SemanticKITTI. The folder structure must be:
```
/path/to/datasets/
|
|- nuscenes/
|  |- lidarseg/
|  | ...
|  |- v1.0-trainval
|
|- semantic_kitti/
|  |- dataset/
```

For WOD, you must preprocess it as described in [Pointcept](https://github.com/Pointcept/Pointcept#waymo)
The folder structure will be:
```
/path/to/datasets/
|
|- waymo_open_dataset_v_1_4_3_processed/
|  |- training/
|  |  |-segment-***
|
|  |- validation/
|  |  |-segment-***
```

## Available models

We provide the following models:

| Model    | Dataset       | Link  |
|----------|---------------|-------|
| VaViT-B  | nuScenes      |       |
| VaViT-B  | SemanticKITTI |       |
| VaViT-B  | WOD           |       |


The nuScenes models are released under the following [terms](https://www.nuscenes.org/terms-of-use).\
The SemanticKITTI are released under the following [terms](https://semantic-kitti.org/dataset.html#licence).\
The WOD models are released under the following [terms](https://waymo.com/open/terms/).

## Evaluation

### nuScenes

To evaluate the provided checkpoint:
```
config="VaViT_B-drop_0.3"
dataset="nuscenes"
PATH_DATASETS=/path/to/datasets

python train.py \
--dataset ${dataset} \
--path_dataset ${PATH_DATASETS}/${dataset}/ \
--log_path ./checkpoints/${dataset}/${config}/ \
--config configs/${dataset}/${config}.yaml \
--gpu 0 \
--restart \
--eval last
```

You can adjust the batch size in the config file `configs/${dataset}/${config}.yaml` if needed.

Replace `--eval last` by `--eval best` to re-evaluate the best checkpoint.

You can also save the predictions on disk and use the official nuScenes devkit for evaluation.
```
config="VaViT_B-drop_0.3"
dataset="nuscenes"
PATH_DATASETS=/path/to/datasets/

python save_nuscenes_predictions.py \
--path_dataset ${PATH_DATASETS}/${dataset}/ \
--config configs/${dataset}/${config}.yaml \
--ckpt ./checkpoints/${dataset}/${config}/ckpt_last.pth \
--result_folder ./predictions/${dataset}/${config}

pip install nuscenes-devkit
git clone https://github.com/nutonomy/nuscenes-devkit.git

python nuscenes-devkit/python-sdk/nuscenes/eval/lidarseg/evaluate.py \
--result_path ./predictions/${dataset}/${config} \
--eval_set val \
--version v1.0-trainval \
--dataroot ${PATH_DATASETS}/${dataset}/ \
--verbose True
```

### SemanticKITTI

To evaluate the last checkpoint:
```
config="VaViT_B-drop_0.5"
dataset="semantic_kitti"
PATH_DATASETS=/path/to/datasets/

python train.py \
--dataset ${dataset} \
--path_dataset ${PATH_DATASETS}/${dataset}/ \
--log_path ./checkpoints/${dataset}/${config}/ \
--config configs/${dataset}/${config}.yaml \
--restart \
--eval last
```

You can adjust the batch size in the config file `configs/${dataset}/${config}.yaml` if needed.

Replace `--eval last` by `--eval best` to re-evaluate the best checkpoint.

**Remark:** *On SemanticKITTI, the code above will extract object instances on the train set (despite this being not necessary for validation). This step can be bypassed by editing the `yaml` config file and changing the entry `instance_cutmix` to `False`. The instances are saved automatically in `/tmp/kitti_instances/`. Do not forget to enable again this augmentation to train a new model on SemanticKITTI.*

You can also save the predictions on disk and use the official SemanticAPI for evaluation.
```
config="VaViT_B-drop_0.5"
dataset="semantic_kitti"
PATH_DATASETS=/path/to/datasets/

python save_semkitti_predictions.py \
--path_dataset ${PATH_DATASETS}/${dataset}/ \
--config configs/${dataset}/${config}.yaml \
--ckpt ./checkpoints/${dataset}/${config}/ckpt_last.pth \
--result_folder ./predictions/${dataset}/${config}/

git clone https://github.com/PRBonn/semantic-kitti-api.git
cd semantic-kitti-api/

python evaluate_semantics.py \
--dataset ${PATH_DATASETS}/${dataset}/dataset \
--predictions ../predictions/${dataset}/${config}/ \
--split valid
```

### WOD

To evaluate the last checkpoint:
```
config="VaViT_B-drop_0.3"
dataset="waymo"
PATH_DATASETS=/path/to/datasets/

python train.py \
--dataset ${dataset} \
--path_dataset ${PATH_DATASETS}/waymo_open_dataset_v_1_4_3_processed/ \
--log_path ./checkpoints/${dataset}/${config}/ \
--config configs/${dataset}/${config}.yaml \
--restart \
--eval last
```

You can adjust the batch size in the config file `configs/${dataset}/${config}.yaml` if needed.

Replace `--eval last` by `--eval best` to re-evaluate the best checkpoint.

## Training

### nuScenes

To retrain VaViT-B on nuScenes:
```
config="VaViT_B-drop_0.3"
dataset="nuscenes"
PATH_DATASETS=/path/to/datasets/

python train.py \
--dataset ${dataset} \
--path_dataset ${PATH_DATASETS}/${dataset}/ \
--log_path ./logs/${dataset}/${config}/ \
--config configs/${dataset}/${config}.yaml \
--gpu 0
```

**Remarks**: 
- *We trained our networks in `bfloat16` (option activated by default). `float16` is also available with the flag `--fp16` but we often ran into numerical instabilities with this option.*
- The final model can be re-evaluated by adding the flags `--restart --eval last`.
- *For multi-GPU training (on one node), replace the flag `--gpu 0` by `--multiprocessing-distributed`.*


### SemanticKITTI

To retrain VaViT-B on SemanticKITTI:
```
config="VaViT_B-drop_0.5"
dataset="semantic_kitti"
PATH_DATASETS=/path/to/datasets/

python train.py \
--dataset ${dataset} \
--path_dataset ${PATH_DATASETS}/${dataset}/ \
--log_path ./logs/${dataset}/${config}/ \
--config configs/${dataset}/${config}.yaml \
--multiprocessing-distributed
```

At the beginning of the training, the instances for cutmix augmentation are saved in `/tmp/kitti_instances/`. *If this process is interrupted before completion, please delete `/tmp/kitti_instances/` and relaunch the training.* You can disable the instance cutmix augmentations by editing the `yaml` config file to set `instance_cutmix` to `False`.


### WOD

To retrain VaViT-B on WOD:
```
config="VaViT_B-drop_0.3"
dataset="waymo"
PATH_DATASETS=/path/to/datasets/

python train.py \
--dataset ${dataset} \
--path_dataset ${PATH_DATASETS}/waymo_open_dataset_v_1_4_3_processed/ \
--log_path ./logs/${dataset}/${config}/ \
--config configs/${dataset}/${config}.yaml \
--multiprocessing-distributed
```

## Acknowledgements
We thank the authors of
```
@inproceedings{berman18lovasz,
  title = {The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure in Neural Networks},
  author = {Berman, Maxim and Triki, Amal Rannen and Blaschko, Matthew B.},
  booktitle = {CVPR},
  year = {2018},
}
```
for making their [implementation](https://github.com/bermanmaxim/LovaszSoftmax) of the Lovász loss publicly available.

## License
VaViT is released under the [Apache 2.0 license](./LICENSE).

The implementation of the Lovász loss in `utils/lovasz.py` is released under [MIT Licence](https://github.com/bermanmaxim/LovaszSoftmax/blob/master/LICENSE).

The SemanticKITTI config file in `datasets/semantic-kitti.yaml` is released under [MIT License](https://github.com/PRBonn/semantic-kitti-api/blob/master/LICENSE)
