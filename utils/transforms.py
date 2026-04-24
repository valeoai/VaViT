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


class Compose:
    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, pcloud, labels):
        for t in self.transformations:
            pcloud, labels = t(pcloud, labels)
        return pcloud, labels


class RandomApply:
    def __init__(self, transformation, prob=0.5):
        self.prob = prob
        self.transformation = transformation

    def __call__(self, pcloud, labels):
        if torch.rand(1).item() < self.prob:
            pcloud, labels = self.transformation(pcloud, labels)
        return pcloud, labels


class Transformation:
    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, pcloud, labels):
        if labels is None:
            return (
                (pcloud, labels)
                if self.inplace
                else (np.array(pcloud, copy=True), labels)
            )

        out = (
            (pcloud, labels)
            if self.inplace
            else (np.array(pcloud, copy=True), np.array(labels, copy=True))
        )
        return out


class Rotation(Transformation):
    def __init__(self, dim=2, range=np.pi, inplace=False):
        super().__init__(inplace)
        self.range = range
        self.inplace = inplace
        if dim == 2:
            self.dims = (0, 1)
        elif dim == 1:
            self.dims = (0, 2)
        elif dim == 0:
            self.dims = (1, 2)
        elif dim == 6:
            self.dims = (4, 5)

    def __call__(self, pcloud, labels):
        # Build rotation matrix
        theta = (2 * torch.rand(1).item() - 1) * self.range
        # Build rotation matrix
        rot = np.array(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        # Apply rotation
        pcloud, labels = super().__call__(pcloud, labels)
        pcloud[:, self.dims] = pcloud[:, self.dims] @ rot
        return pcloud, labels


class Scale(Transformation):
    def __init__(self, dims=(0, 1), range=0.05, inplace=False):
        super().__init__(inplace)
        self.dims = dims
        self.range = range

    def __call__(self, pcloud, labels):
        pcloud, labels = super().__call__(pcloud, labels)
        scale = 1 + (2 * torch.rand(1).item() - 1) * self.range
        pcloud[:, self.dims] *= scale
        return pcloud, labels


class Flip(Transformation):
    def __init__(self, axes=(1, 2), inplace=False):
        super().__init__(inplace=inplace)
        self.axes = axes

    def __call__(self, pcloud, labels):
        pcloud, labels = super().__call__(pcloud, labels)
        id = torch.randint(len(self.axes), (1,))[0]
        pcloud[:, self.axes[id]] *= -1.0
        return pcloud, labels


class Voxelize(Transformation):
    def __init__(self, dims=(0, 1, 2), voxel_size=0.1, random=False):
        super().__init__(inplace=True)
        self.dims = dims
        self.voxel_size = voxel_size
        self.random = random
        assert voxel_size >= 0

    def __call__(self, pcloud, labels):
        pc, labels = super().__call__(pcloud, labels)
        if self.voxel_size <= 0:
            return pc, labels

        if self.random:
            permute = torch.randperm(pc.shape[0])
            pc, labels = pc[permute], labels[permute]

        pc_shift = pc[:, self.dims] - pc[:, self.dims].min(0, keepdims=True)

        _, ind = np.unique(
            (pc_shift / self.voxel_size).astype("int"), return_index=True, axis=0
        )

        return pc[ind, :], None if labels is None else labels[ind]


class PillarMix(Transformation):
    def __init__(self, side_length=5.0, num_pcs=3, size_mem=10, keep_prob=0.5):
        super().__init__(inplace=True)
        self.num_pcs = num_pcs - 1
        self.side_length = side_length
        self.size_mem = size_mem
        self.keep_prob = keep_prob
        self.mem = []

    def sample_pcs(self, pc, label):
        pcs, labels = [pc], [label]
        if self.num_pcs == 0:
            return pcs, labels
        # Pick some point cloud from memory
        for idx in torch.randint(len(self.mem), (self.num_pcs,)):
            pcs.append(self.mem[idx][0])
            labels.append(self.mem[idx][1])
        # Keep current point cloud in memory (or not)
        if torch.rand(1).item() < self.keep_prob:
            if len(self.mem) == self.size_mem:
                self.mem.pop(0)
            self.mem.append((pc, label))
        return pcs, labels

    def __call__(self, pc, label):

        # Not enough point cloud in memory yet
        if len(self.mem) < self.num_pcs:
            self.mem.append((pc, label))
            return pc, label

        # Sample point clouds for augmentations
        # Base / current point cloud is the first in the list
        pcs, labels = self.sample_pcs(pc, label)

        # Find occupied pillars for each point cloud
        cloud_square_maps = []
        cloud_point_inverses = []
        for p_cloud in pcs:
            grid_coords = np.floor(p_cloud[:, :2] / self.side_length).astype(np.int32)
            unique_sq, inverse = np.unique(grid_coords, axis=0, return_inverse=True)
            cloud_point_inverses.append(inverse)
            # Map (x, y) -> local_index
            sq_map = {tuple(sq): idx for idx, sq in enumerate(unique_sq)}
            cloud_square_maps.append(sq_map)

        # Occupied pillar of base point cloud
        base_sq_map = cloud_square_maps[0]

        # Initialize masks: Base point cloud starts with all True, others start with all False
        keep_square_masks = [np.zeros(len(m), dtype=bool) for m in cloud_square_maps]
        keep_square_masks[0][:] = True

        # Loop over occupied pillar of base point cloud
        for sq, base_local_idx in base_sq_map.items():
            # Check which other clouds also have points in this specific square
            candidates = [i for i in range(1, len(pcs)) if sq in cloud_square_maps[i]]

            if len(candidates) > 0:
                # Pick a point cloud for this pillar
                idx = torch.randint(len(candidates) + 1, (1,)).item()

                # Base point cloud selected
                if idx == 0:
                    continue

                # Selected point cloud
                winner_idx = candidates[idx - 1]

                # Turn OFF the base point cloud pillar
                keep_square_masks[0][base_local_idx] = False

                # Turn ON the selected point cloud pillar
                other_local_idx = cloud_square_maps[winner_idx][sq]
                keep_square_masks[winner_idx][other_local_idx] = True

        # Reconstruct point cloud
        final_pc, final_labels = [], []
        for i in range(len(pcs)):
            point_mask = keep_square_masks[i][cloud_point_inverses[i]]
            if point_mask.any():
                final_pc.append(pcs[i][point_mask])
                final_labels.append(labels[i][point_mask])

        return np.concatenate(final_pc, axis=0), np.concatenate(final_labels, axis=0)
