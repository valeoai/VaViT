# Copyright 2026 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import random
import warnings

import numpy as np
import torch
import yaml

import utils.transforms as tr
from datasets import LIST_DATASETS, Collate
from utils.metrics import SemSegLoss
from utils.scheduler import WarmupCosine
from utils.trainer import TrainingManager
from vavit.segmenter import Segmenter


def load_model_config(file):
    with open(file) as f:
        config = yaml.safe_load(f)
    return config


def get_train_augmentations(config, args):
    list_of_transf = []

    print("\nAugmentation")
    for aug_name in config["augmentations"].keys():
        if aug_name == "rotation":
            for d in config["augmentations"]["rotation"][0]:
                print("Rotation", d)
                list_of_transf.append(tr.Rotation(inplace=False, dim=d))
        elif aug_name == "flip":
            for axes in config["augmentations"]["flip"]:
                print("flip", axes)
                list_of_transf.append(
                    tr.RandomApply(tr.Flip(axes=tuple(axes), inplace=False), prob=0.66)
                )
        elif aug_name == "scale":
            scale = config["augmentations"]["scale"][-1]
            for dims in config["augmentations"]["scale"][:-1]:
                print("scale", dims, scale)
                list_of_transf.append(tr.Scale(inplace=False, dims=dims, range=scale))
        elif aug_name == "pillarmix":
            pillarmix = tr.PillarMix(
                side_length=config["augmentations"]["pillarmix"][0],
                num_pcs=config["augmentations"]["pillarmix"][1],
                size_mem=config["augmentations"]["pillarmix"][2],
            )
            list_of_transf.append(pillarmix)
        elif aug_name in ["instance_cutmix"]:
            # Do nothing here, directly handled in semantic kitti dataset
            continue
        else:
            raise ValueError(f"Unknown augmentation: {aug_name}")

    print(list_of_transf)
    list_of_transf = tr.Compose(list_of_transf)

    return list_of_transf


def get_datasets(config, args):
    # Shared parameters
    kwargs = {
        "rootdir": args.path_dataset,
        "input_feat": config["embedding"]["input_feat"],
        "voxel_size": config["embedding"]["voxel_size"],
        "num_neighbors_emb": config["embedding"]["neighbors"],
        "bev_size": config["vit"]["bev_size"],
    }

    # Get datatset
    DATASET = LIST_DATASETS.get(args.dataset.lower())
    if DATASET is None:
        raise ValueError(f"Dataset {args.dataset.lower()} not available.")

    # Train dataset
    train_augmentations = get_train_augmentations(config, args)
    train_dataset = DATASET(
        phase="train",
        train_augmentations=train_augmentations,
        instance_cutmix=config["augmentations"].get("instance_cutmix", False),
        **kwargs,
    )

    # Validation dataset
    val_dataset = DATASET(
        phase="val",
        **kwargs,
    )

    return train_dataset, val_dataset, train_augmentations


def get_dataloader(train_dataset, val_dataset, args):
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=Collate(),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=Collate(),
    )

    return train_loader, val_loader, train_sampler


def get_optimizer(model, config):
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config["optim"]["lr"],
        weight_decay=config["optim"]["weight_decay"],
    )
    return optim


def get_scheduler(optimizer, config, len_train_loader):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        WarmupCosine(
            config["scheduler"]["epoch_warmup"] * len_train_loader,
            config["scheduler"]["max_epoch"] * len_train_loader,
            config["scheduler"]["min_lr"] / config["optim"]["lr"],
        ),
    )
    return scheduler


def distributed_training(gpu, ngpus_per_node, args, config):

    # --- Init. distributing training
    args.gpu = gpu
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # --- Build network
    args_model = {
        "emb_cin": config["embedding"]["cin"],
        "emb_chidden": config["embedding"]["chidden"],
        "emb_num_layers": config["embedding"]["num_layers"],
        "vit_dim": config["vit"]["dim"],
        "vit_depth": config["vit"]["depth"],
        "vit_num_heads": config["vit"]["num_heads"],
        "vit_expansion": config["vit"]["expansion"],
        "merge_chidden": config["merge_head"]["chidden"],
        "merge_cout": config["merge_head"]["cout"],
        "nb_class": config["classif"]["nb_class"],
        "drop_path": config["vit"]["drop_prob"],
    }
    model = Segmenter(**args_model)

    # ---
    args.batch_size = config["dataloader"]["batch_size"]
    args.workers = config["dataloader"]["num_workers"]
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs of the current node.
        args.batch_size = int(config["dataloader"]["batch_size"] / ngpus_per_node)
        args.workers = int(
            (config["dataloader"]["num_workers"] + ngpus_per_node - 1) / ngpus_per_node
        )
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
    elif args.gpu is not None:
        # Training on one GPU
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    if args.gpu == 0 or args.gpu is None:
        print(f"Model:\n{model}")
        nb_param = sum([p.numel() for p in model.parameters()]) / 1e6
        print(f"{nb_param} x 10^6 trainable parameters ")

    # --- Optimizer
    optim = get_optimizer(model, config)

    # --- Dataset
    train_dataset, val_dataset, train_augmentations = get_datasets(config, args)
    train_loader, val_loader, train_sampler = get_dataloader(
        train_dataset, val_dataset, args
    )

    # --- Loss function
    loss = SemSegLoss(
        lovasz_weight=config["loss"]["lovasz"],
    )
    loss = loss.cuda(args.gpu)

    # --- Scheduler
    scheduler = get_scheduler(optim, config, len(train_loader))

    # --- Training
    mng = TrainingManager(
        model,
        loss,
        train_loader,
        val_loader,
        train_sampler,
        optim,
        scheduler,
        config["scheduler"]["max_epoch"],
        args.log_path,
        args.gpu,
        args.world_size,
        args.fp16,
        LIST_DATASETS.get(args.dataset.lower()).CLASS_NAME,
        tensorboard=(not args.eval),
    )
    if args.restart or args.eval in ["last", "best"]:
        mng.load_state(best=(args.eval == "best"))
    if args.eval in ["last", "best"]:
        mng.one_epoch(training=False)
    else:
        mng.train()


def main(args, config):
    # --- Fixed args
    # Device
    args.device = "cuda"
    # Node rank for distributed training
    args.rank = 0
    # Number of nodes for distributed training'
    args.world_size = 1
    # URL used to set up distributed training
    args.dist_url = "tcp://127.0.0.1:4444"
    # Distributed backend'
    args.dist_backend = "nccl"
    # Distributed processing
    args.distributed = args.multiprocessing_distributed

    # Create log directory
    os.makedirs(args.log_path, exist_ok=True)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)

    if args.gpu is not None:
        args.gpu = 0
        args.distributed = False
        args.multiprocessing_distributed = False
        warnings.warn(
            "You have chosen a specific GPU. This will completely disable data parallelism."
        )

    # Extract instances for cutmix
    if "instance_cutmix" in config["augmentations"].keys():
        get_datasets(config, args)

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        torch.multiprocessing.spawn(
            distributed_training,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, config),
        )
    else:
        # Simply call main_worker function
        distributed_training(args.gpu, ngpus_per_node, args, config)


def get_default_parser():

    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset",
        default="nuscenes",
    )
    parser.add_argument(
        "--path_dataset",
        type=str,
        help="Path to dataset",
        default="/datasets_local/nuscenes/",
    )
    parser.add_argument(
        "--log_path", type=str, required=True, help="Path to log folder"
    )
    parser.add_argument(
        "--restart", action="store_true", default=False, help="Restart training"
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="Seed for initializing training"
    )
    parser.add_argument(
        "--gpu", default=None, type=int, help="Set to any number to use gpu 0"
    )
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enable autocast for mix precision training",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config"
    )
    parser.add_argument(
        "--eval",
        default="",
        help="Run validation only",
    )

    return parser


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()
    config = load_model_config(args.config)
    main(args, config)
