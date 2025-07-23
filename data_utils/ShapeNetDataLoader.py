# *_*coding:utf-8 *_*
import json
import os
import warnings
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


def pc_normalize(pc):
    """Normalize 2D array of x-y-z"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class PartNormalDataset(Dataset):
    """Dataset class for ShapeNet data.

    Attributes
    ----------
    npoints: int
        Set to 2048
    cache: dict[int, tuple[np.ndarray, int, int]]
        Also see __getitem__(); it's not used wtf.
        {index: (point_set, cls, seg),...}
        {
            index: (
                numpy.ndarray, # x-y-z(-normal) * npoints
                cls,           # <classes["Airplane"]>
                seg,           # label
            ),... # Up to length <cache_size>=20000
        }
    cache_size: int
        Max length of <cache>; 200000.
    cat: dict[str, str]
        <catfile> -> dict
        Only includes __init__() param <class_choice>.
        {'Airplane': 02691156, ...}
    catfile: pathlib.Path
        <root>/synsetoffset2category.txt
        data/shapenetcore_partanno_segmentation_benchmark_v0_normal/synsetoffset2category.txt
    classes: dict[str, int]
        {<cat.keys()>: <classes_original.values()>}
        {'Airplane': 0, 'Bag': 1, ...}
    classes_original: dict[str, int]
        <cat> -> range()
        {'Airplane': 0, 'Bag': 1, ...}
    datapath: list[tuple[str, os.PathLike]]
        [
            ('Airplane', <root>/02691156/xxxxxx.txt),...
            ('Bag', <root>/02773838/xxxxxx.txt),...
            # all txt files
        ]
    meta: dict[str, list[os.PathLike]]
        train_test_split/*.json -> dict
        {
            'Airplane': [
                # list of txt files
                <root>/02691156/xxxxxx.txt,...
            ],...
        }
    normal_channel: bool
        - True: x-y-z-nx-ny-nz-label
        - False: x-y-z-label
    root: os.PathLike
        Root directory of dataset. data/shapenetcore_partanno_segmentation_benchmark_v0_normal.

    Notes
    -----
    How category propagates:
    catfile
        -> cat    <- __init__(class_choice)
            -> classes_original
            ->   -> classes
                    -> cache

    How data path propagates:
    root
        -> catfile (synsetoffset2category.txt)
        -> meta    (.txt data files)
            -> datapath

    """

    def __init__(
        self, root, npoints, split="train", class_choice=None, normal_channel=False
    ):
        """load shapenet dataset.

        Args:
            root : Root directory of dataset. data/shapenetcore_partanno_segmentation_benchmark_v0_normal.
            npoints : Number of points. Default config is 2048.
            split : Which split file(<root>/traintest_split/*.json) to be used Defaults to 'train'.
                Options are:
                - train : training set
                - val : validation set
                - test : test set
                - trainval : training and validation set
            class_choice : Write a list of categories that you would only want to use. Defaults to None.
            normal_channel : If to use normals. Defaults to False.
        """
        self.npoints = npoints
        self.root = root
        self.catfile = (
            Path(root) / "synsetoffset2category.txt"
        )  # category names -> folder names
        self.cat = {}
        self.normal_channel = normal_channel

        with open(self.catfile, "r") as f:
            for line in f:
                ls = line.strip().split()  # ['Airplane', '02691156']
                self.cat[ls[0]] = ls[1]  # {'Airplane': '02691156', ...}
        self.cat = {
            k: v for k, v in self.cat.items()
        }  # maybe because self.cat is not guaranteed to be a dict??
        self.classes_original = dict(
            zip(self.cat, range(len(self.cat)))
        )  # {'Airplane': 0, 'Bag': 1, ...}

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(
            Path(root) / "train_test_split" / "shuffled_train_file_list.json", "r"
        ) as f:
            train_ids = set([str(d.split("/")[2]) for d in json.load(f)])
        with open(
            Path(root) / "train_test_split" / "shuffled_val_file_list.json", "r"
        ) as f:
            val_ids = set([str(d.split("/")[2]) for d in json.load(f)])
        with open(
            Path(root) / "train_test_split" / "shuffled_test_file_list.json", "r"
        ) as f:
            test_ids = set([str(d.split("/")[2]) for d in json.load(f)])
        for item in self.cat:  # item = 'Airplane', 'Bag', ...
            # print('category', item)
            self.meta[item] = []
            dir_point = Path(root) / self.cat[item]  # <root>/02691156
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4]) # -4 is to remove the '.txt' extension
            if split == "trainval":
                fns = [
                    fn
                    for fn in fns
                    if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))
                ]
            elif split == "train":
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == "val":
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == "test":
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print("Unknown split: %s. Exiting.." % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = os.path.splitext(os.path.basename(fn))[0]
                self.meta[item].append(os.path.join(dir_point, token + ".txt"))

        self.datapath = []
        # [('Airplane', <root>/02691156/xxxxxx.txt), ('Bag', <root>/02773838/xxxxxx.txt), ...]
        for item in self.cat:  # item = 'Airplane', 'Bag', ...
            for fn in self.meta[item]:  # fn = <root>/02691156/xxxxxx.txt
                self.datapath.append((item, fn))

        # Needed if the argument class_choice is provided
        self.classes = {}
        for item in self.cat:
            self.classes[item] = self.classes_original[item]
        # Default segmentation classes
        # self.seg_classes = {
        #     # category  : list of labels
        #     'Earphone'  : [16, 17, 18],
        #     'Motorbike' : [30, 31, 32, 33, 34, 35],
        #     'Rocket'    : [41, 42, 43],
        #     'Car'       : [8, 9, 10, 11],
        #     'Laptop'    : [28, 29],
        #     'Cap'       : [6, 7],
        #     'Skateboard': [44, 45, 46],
        #     'Mug'       : [36, 37],
        #     'Guitar'    : [19, 20, 21],
        #     'Bag'       : [4, 5],
        #     'Lamp'      : [24, 25, 26, 27],
        #     'Table'     : [47, 48, 49],
        #     'Airplane'  : [0, 1, 2, 3],
        #     'Pistol'    : [38, 39, 40],
        #     'Chair'     : [12, 13, 14, 15],
        #     'Knife'     : [22, 23]
        # }

        self.cache = {}  # {int[index]: tuple[point_set, cls, seg],...}
        self.cache_size = 20000

    def __getitem__(self, index):
        """Read file, random sample npoints and return."""
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]  # ('Airplane', <root>/02691156/xxxxxx.txt)
            cat = self.datapath[index][0]  # 'Airplane'
            # self.data[:, :, index] comes from the same file;
            # cls (category) should be the same value for a given index
            cls = self.classes[cat]  # 0
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)  # <root>/02691156/xxxxxx.txt
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(
            point_set[:, 0:3]
        )  # normalize to point cloud centroid

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg  # -> torch[B*N*C], torch[B*1], torch[B*N]
        # point_set: np[npoints*3|6] = points
        # cls: np[1] = category index self.classes["Airplane"]
        # seg:  np[npoints] = label

    def __len__(self):
        """# of txt files."""
        return len(self.datapath)
