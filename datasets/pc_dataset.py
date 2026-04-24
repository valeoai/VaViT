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

import numpy as np
import torch
from scipy.spatial import cKDTree as KDTree
from torch.utils.data import Dataset

import utils.transforms as tr


class PCDataset(Dataset):
    def __init__(
        self,
        rootdir=None,
        phase="train",
        input_feat="intensity",
        voxel_size=0.1,
        train_augmentations=None,
        num_neighbors_emb=16,
        bev_size=0.5,
        instance_cutmix=False,
        num_instances=40,
    ):
        super().__init__()

        # Dataset split
        self.phase = phase
        assert self.phase in ["train", "val", "trainval", "test"]

        # Root directory of dataset
        self.rootdir = rootdir

        # Input features to compute for each point
        self.input_feat = input_feat

        # Downsample input point cloud by small voxelization
        self.downsample = tr.Voxelize(
            dims=(0, 1, 2),
            voxel_size=voxel_size,
            random=(self.phase == "train" or self.phase == "trainval"),
        )

        # Number of neighbors for embedding layer
        self.num_neighbors_emb = num_neighbors_emb

        # BEV size
        self.bev_size = bev_size

        # Train time augmentations
        if train_augmentations is not None:
            assert self.phase in ["train", "trainval"]
        self.train_augmentations = train_augmentations

        # Flag for instance cutmix
        self.instance_cutmix = instance_cutmix
        self.num_instances = num_instances

    def prepare_input_features(self, pc_orig):
        # Concatenate desired input features to coordinates
        pc = [pc_orig[:, :3]]  # Initialize with coordinates
        for type in self.input_feat:
            if type == "intensity":
                pc.append(pc_orig[:, 3:])
            elif type == "radius":
                r_xyz = np.linalg.norm(pc_orig[:, :3], axis=1, keepdims=True)
                pc.append(r_xyz)
            elif type == "xyz":
                pc.append(pc_orig[:, :3])
            else:
                raise ValueError(f"Unknown feature: {type}")
        return np.concatenate(pc, 1)

    def get_bev_idx(self, pc):
        # Quantize coordinates in BEV
        pc_quant = np.floor(pc[:, :2] / self.bev_size).astype(np.int32)

        # Get pillar indices
        unique_coords, bev_idx = np.unique(pc_quant, axis=0, return_inverse=True)

        # xy offset
        pillar_centers = (unique_coords.astype(np.float32) * self.bev_size) + (
            self.bev_size / 2
        )

        return bev_idx, pillar_centers

    def load_pc(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        # Load original point cloud
        pc_orig, labels_orig, filename = self.load_pc(index)

        # Prepare input feature
        pc_orig = self.prepare_input_features(pc_orig)

        # Voxel downsampling
        pc, labels = self.downsample(pc_orig, labels_orig)

        # Augment data
        if self.train_augmentations is not None:
            pc, labels = self.train_augmentations(pc, labels)

        # Map points to BEV
        idx_bev, xy_bev = self.get_bev_idx(pc)

        # Get neighbors for point embedding layer
        assert pc.shape[0] > (self.num_neighbors_emb + 1)
        kdtree = KDTree(pc[:, :3])
        neighbors = kdtree.query(pc[:, :3], k=self.num_neighbors_emb + 1)[1]
        neighbors = neighbors[:, 1:]  # Remove self-loop

        # Nearest neighbor interpolation to undo voxelisation at validation time
        if self.phase in ["train", "trainval"]:
            upsample = np.arange(pc.shape[0])
        else:
            upsample = kdtree.query(pc_orig[:, :3], k=1)[1]

        # Output to return
        out = (
            # Point features
            pc[:, 3:],
            # Point labels of original entire point cloud
            labels if self.phase in ["train", "trainval"] else labels_orig,
            # Neighborhood for point embedding layer, which provides tokens to waffleiron backbone
            neighbors,
            # Contiguous indices for BEV projection
            idx_bev,
            # Coordinates of pillars in bev
            xy_bev,
            # For interpolation from voxelized & cropped point cloud to original point cloud
            upsample,
            # Filename of original point cloud
            filename,
        )

        return out


class Collate:
    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index

    def __call__(self, list_data):
        # Extract all data
        list_of_data = (list(data) for data in zip(*list_data))
        feat, labels, neighbors, idx_bev, xy_bev, upsample, filename = list_of_data

        # Modify bev indices with zero padding to allow reshaping to (batch_size, seq_len, dim_feat)
        idx_occ, idx_pad = [], []
        Npad = max([i.max() + 1 for i in idx_bev])
        for i in range(len(idx_bev)):
            idx_bev[i] += i * Npad
            largest_occupied = idx_bev[i].max()
            idx_occ.append(np.arange(i * Npad, largest_occupied + 1).astype(np.int32))
            if (largest_occupied + 1) < (i + 1) * Npad:
                idx_pad.append(
                    np.arange(largest_occupied + 1, (i + 1) * Npad).astype(np.int32)
                )
        splits_bev = [Npad for _ in range(len(idx_bev))]
        idx_bev = torch.from_numpy(np.concatenate(idx_bev, 0))  # Npc
        xy_bev = torch.from_numpy(np.concatenate(xy_bev, 0))  # Npc
        idx_occ = torch.from_numpy(np.concatenate(idx_occ, 0))
        if len(idx_pad) > 0:
            idx_pad = torch.from_numpy(np.concatenate(idx_pad, 0))
        else:
            idx_pad = None

        # End of each point sequences
        splits_pc = np.cumsum([len(f) for f in feat])
        assert len(splits_pc) == len(splits_bev)

        # Concatenate along sequence dimension
        feat = torch.from_numpy(np.concatenate(feat, 0))  # Npc x C
        labels = torch.from_numpy(np.concatenate(labels, 0))  # Norig or Npc
        assert len(feat) == splits_pc[-1]

        # Adapt other indices
        for i in range(1, len(splits_pc)):
            neighbors[i] += splits_pc[i - 1]
            upsample[i] += splits_pc[i - 1]
        neighbors = torch.from_numpy(np.concatenate(neighbors, 0))  # Npc x K
        upsample = torch.from_numpy(np.concatenate(upsample, 0))  # Norig or Npc
        assert len(neighbors) == splits_pc[-1]
        assert len(labels) == len(upsample)
        assert len(idx_bev) == splits_pc[-1]

        # Prepare output variables
        out = {
            "feat": feat.float(),
            "neighbors": neighbors.int(),
            "idx_bev": idx_bev.long(),
            "xy_bev": xy_bev.float(),
            "idx_occ": idx_occ.int(),
            "idx_pad": None if idx_pad is None else idx_pad.int(),
            "splits_bev": torch.tensor(splits_bev).int(),
            "upsample": upsample.int(),
            "labels": labels.long(),
            "filename": filename,
        }

        return out
