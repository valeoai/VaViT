import glob
import os

import numpy as np

from .pc_dataset import PCDataset


class WaymoSemSeg(PCDataset):
    CLASS_NAME = [
        [
            "car",
            "truck",
            "bus",
            "other_vehicle",
            "motorcyclist",
            "bicyclist",
            "pedestrian",
            "sign",
            "traffic_light",
            "pole",
            "construction_cone",
            "bicycle",
            "motorcycle",
            "building",
            "vegetation",
            "tree_trunk",
            "curb",
            "road",
            "lane_marker",
            "other_ground",
            "walkable",
            "sidewalk",
        ]
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.files = []
        if self.phase == "train":
            for file in glob.glob(os.path.join(self.rootdir, "training", "*", "*")):
                self.files.append(file)
            assert len(self.files) == 23691
        elif self.phase == "val":
            for file in glob.glob(os.path.join(self.rootdir, "validation", "*", "*")):
                self.files.append(file)
            assert len(self.files) == 5976
        else:
            raise ValueError
        self.files = np.sort(self.files)

        assert not self.instance_cutmix, "Instance CutMix not implemented on Waymo"

    def __len__(self):
        return len(self.files)

    def load_pc(self, index):
        # Load point cloud
        coord = np.load(self.files[index] + "/coord.npy")
        strength = np.load(self.files[index] + "/strength.npy")
        pc = np.concatenate((coord, strength), axis=1)

        # Extract Label
        labels = np.load(self.files[index] + "/segment.npy").astype(np.int32)
        labels[labels == -1] = 255

        return pc, labels, self.files[index]
