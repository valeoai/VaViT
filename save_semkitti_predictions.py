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

import numpy as np
import torch
import yaml
from tqdm import tqdm

from datasets import Collate, SemanticKITTI
from vavit import Segmenter

if __name__ == "__main__":
    # --- Arguments
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--path_dataset",
        type=str,
        help="Path to SemanticKITTI dataset",
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        help="Path to where result folder",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--phase",
        default="val",
        help="val or test",
    )
    args = parser.parse_args()

    # ---
    os.makedirs(args.result_folder, exist_ok=True)

    # --- Load network config file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # --- SemanticKITTI config
    with open("./datasets/semantic-kitti.yaml") as stream:
        semkittiyaml = yaml.safe_load(stream)
    remapdict = semkittiyaml["learning_map_inv"]
    maxkey = max(remapdict.keys())
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())

    # --- Dataloader
    dataset = SemanticKITTI(
        rootdir=args.path_dataset,
        input_feat=config["embedding"]["input_feat"],
        voxel_size=config["embedding"]["voxel_size"],
        num_neighbors_emb=config["embedding"]["neighbors"],
        bev_size=config["vit"]["bev_size"],
        phase=args.phase,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=Collate(),
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
    net = Segmenter(**args_model)

    # --- Load weights
    ckpt = torch.load(args.ckpt, map_location="cpu")["net"]
    new_ckpt = {}
    for key in ckpt.keys():
        if key.startswith("module."):
            new_ckpt[key[len("module.") :]] = ckpt[key]
        else:
            new_ckpt[key] = ckpt[key]
    net.load_state_dict(new_ckpt, strict=True)
    net = net.cuda()
    net.eval()

    # --- Evaluation
    for it, batch in enumerate(
        tqdm(loader, bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}")
    ):
        # Network inputs
        feat = batch["feat"].cuda(0, non_blocking=True)
        neighbors_emb = batch["neighbors"].cuda(0, non_blocking=True)
        idx_to_bev = batch["idx_bev"].cuda(0, non_blocking=True)
        xy_bev = batch["xy_bev"].cuda(0, non_blocking=True)
        idx_occ = batch["idx_occ"].cuda(0, non_blocking=True)
        if batch["idx_pad"] is not None:
            idx_pad = batch["idx_pad"].cuda(0, non_blocking=True)
        else:
            idx_pad = None
        bev_splits = batch["splits_bev"].cuda(0, non_blocking=True)
        labels = batch["labels"].cuda(0, non_blocking=True)
        upsample = batch["upsample"].cuda(0, non_blocking=True)
        net_inputs = (
            feat,
            neighbors_emb,
            xy_bev,
            idx_to_bev,
            idx_occ,
            idx_pad,
            bev_splits,
        )

        # Get prediction
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.inference_mode():
                vote = net(*net_inputs)[upsample]

        # Save prediction
        pred_label = vote.max(1)[1] + 1  # Shift by 1 because of ignore_label at index 0
        label = pred_label.cpu().numpy().reshape(-1).astype(np.uint32)
        upper_half = label >> 16  # get upper half for instances
        lower_half = label & 0xFFFF  # get lower half for semantics
        lower_half = remap_lut[lower_half]  # do the remapping of semantics
        label = (upper_half << 16) + lower_half  # reconstruct full label
        label = label.astype(np.uint32)
        # Save result
        assert batch["filename"][0] == batch["filename"][-1]
        label_file = batch["filename"][0][
            len(os.path.join(dataset.rootdir, "dataset/")) :
        ]
        label_file = label_file.replace("velodyne", "predictions")[:-3] + "label"
        label_file = os.path.join(args.result_folder, label_file)
        os.makedirs(os.path.split(label_file)[0], exist_ok=True)
        label.tofile(label_file)
