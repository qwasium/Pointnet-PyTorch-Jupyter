"""
Author: qwasium
Date: Jul 2025
"""

import argparse
import importlib
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

HERE = Path(__file__).parent


def txt_path_to_batch_tensor(
    path_list: list[os.PathLike],
    npoints: int = 2048,
    normals: bool = False,
    label: bool = True,
) -> np.ndarray:
    """[xxx.txt, yyy.txt, zzz.txt,...] -> np[npoint * xyz-(norm)-(label) * batch]"""
    n_channels = 3
    if normals:
        n_channels += 3
    if label:
        n_channels += 1
    batch_tensor = np.zeros([npoints, n_channels, len(path_list)])
    for b, txt_path in enumerate(path_list):
        assert Path(txt_path).exists(), f"Data doesn't exist: {txt_path}"
        point_ary = np.loadtxt(txt_path).astype(np.float32)
        point_ary = point_ary[
            np.random.choice(point_ary.shape[1], npoints, replace=True), :
        ]
        if not normals:
            point_ary = point_ary[:, [0, 1, 2, 6]]  # x, y, z, label
        batch_tensor[:, :, b] = point_ary
    return batch_tensor  # np[N*C*B]


class AddImportPath:
    """Temporarily add path to sys.path

    Usage
    -----
    with AddImportPath(<module directory>):
        # import stuff
    """

    def __init__(self, mod_dir: os.PathLike):
        mod_dir = Path(mod_dir).resolve()
        assert mod_dir.exists(), f"Doesn't exist: {mod_dir}"
        self.mod_dir = str(mod_dir)

    def __enter__(self):
        sys.path.insert(0, self.mod_dir)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.mod_dir)
        except:
            warnings.warn(f"Failed to remove from sys.path: {self.mod_dir}")
            pass


class PointnetInference:
    def __init__(
        self, config: dict, seg_classes: dict[str, list[int]], log_events: bool = False
    ):
        """

        Parameters
        ----------
        config: dict
            Configuration dictionary.
            Keys & values:
            - model_name: str
                Import model with this name.
                Example: "pointnet2_part_seg_msg"
            - "gpu": int
                GPU id to set for $CUDA_VISIBLE_DEVICES.
            - "num_point": int
                Input size in point numbers. Use 2048.
            - "normal": bool
                True: Data is x-y-z-nx-ny-nz(-label)
                False: Data is x-y-z(-label)
            - "num_votes": int
                Number of votes to use for segmentation.
            - "data_dir": os.PathLike
                Path to data directory.
            - "log_dir": os.PathLike
                Name of log directory.
                Example: log/part_seg/<experiment name>/
            - "pt_path": os.PathLike()
                Path to model weights.
                Example: <log_dir>/checkpoints/best_model.pth
        seg_classes: dict[str, list[int]],
            Shapenet style segmentation classes:
            Key order matters.
                {
                    "Earphone"  : [16, 17, 18],
                    "Motorbike" : [30, 31, 32, 33, 34, 35],
                    "Rocket"    : [41, 42, 43],
                    "Car"       : [8, 9, 10, 11],
                    "Laptop"    : [28, 29],
                    "Cap"       : [6, 7],
                    "Skateboard": [44, 45, 46],
                    "Mug"       : [36, 37],
                    "Guitar"    : [19, 20, 21],
                    "Bag"       : [4, 5],
                    "Lamp"      : [24, 25, 26, 27],
                    "Table"     : [47, 48, 49],
                    "Airplane"  : [0, 1, 2, 3],
                    "Pistol"    : [38, 39, 40],
                    "Chair"     : [12, 13, 14, 15],
                    "Knife"     : [22, 23],
                }
        log_events: bool
            - True: enble logging
            - False: disable logging

        """
        self.config = config
        self.seg_classes = seg_classes
        self.log_events = log_events

        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

        # logging
        self.logger = logging.getLogger("Model")
        if self.log_events:
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler = logging.FileHandler(config["log_dir"] / "eval.txt")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        self.log_string("PARAMETER ...", self.log_events)
        for key, val in self.config.items():
            self.log_string(f"    {key}: {val}")

        label_to_cat: dict[int, str] = {}  # {<label>: <category>,...}
        for cat, labels in self.seg_classes.items():
            for label in labels:
                label_to_cat[label] = cat
        cat_to_idx: dict[str, int] = {
            cat: idx for idx, cat in enumerate(self.seg_classes)
        }
        self.label_to_cat_idx: dict[int, int] = {
            label: cat_to_idx[cat] for label, cat in label_to_cat.items()
        }

        self.n_classes = len(self.seg_classes)  # 16
        self.n_parts = len(
            [
                seg_id
                for seg_val_sublist in self.seg_classes.values()
                for seg_id in seg_val_sublist
            ]
        )  # 50

        # model
        with AddImportPath(HERE / "models"):
            try:
                model = importlib.import_module(self.config["model_name"])
            except ImportError as err:
                self.log_string(
                    f"Failed to load {self.config['model_name']} from pwd: {os.getcwd()}",
                    self.log_events,
                )
                raise err

        self.classifier = model.get_model(
            self.n_parts, normal_channel=self.config["normal"]
        ).cuda()
        checkpoint = torch.load(self.config["pt_path"], weights_only=False)
        self.classifier.load_state_dict(checkpoint["model_state_dict"])

    def log_string(self, log_str: str, log_events: bool = True, std_out: bool = True):
        if std_out:
            print(log_str)
        if log_events:
            self.logger.info(log_str)

    @staticmethod
    def to_categorical(y, num_classes):
        """1-hot encodes a tensor

        Parameters
        ----------
        y: array of int
            Values correspond to category; range of each value is 0:num_classes-1
            Length: batch size
        num_classes: int
            Number of segmentation categories.

        Return
        ------
        new_y: torch.Tensor
            One-hot encoded 2D-tensor.
            Shape: len(y) * num_classes
        """
        new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
        if y.is_cuda:
            return new_y.cuda()
        return new_y

    def main(self, data: np.ndarray):
        """main function

        Parameters
        ----------
        data: np.ndarray
            Data in np[N*C*B]
        """
        if self.config["normal"] and data.shape[1] != 7:
            self.log_string(
                "provided data doesn't have 7 columns, aborting.", self.log_events
            )
            raise ValueError("data should have columns x-y-z-nx-ny-nz-label")
        elif not self.config["normal"] and data.shape[1] != 4:
            self.log_string(
                "provided data doesn't have 4 columns, aborting.", self.log_events
            )
            raise ValueError("data should have columns x-y-z-label")
        assert data.shape[0] == self.config["num_point"], "Number of points is wrong."

        # normalize x-y-z to point cloud centroid
        xyz = data[:, :3, :]  # N*3*B
        centroid = np.mean(xyz, axis=0)  # 3*B
        xyz = xyz - centroid  # N*3*B
        dist = np.sqrt(np.sum(xyz**2, axis=1))  # N*B
        max_dist = np.max(dist, axis=0)  # B
        xyz = xyz / max_dist
        data[:, :3, :] = xyz

        with torch.no_grad():
            classifier = self.classifier.eval()

            points = torch.from_numpy(
                data[:, :-1, :].astype(np.float32).transpose()
            ).cuda()  # N*C*B -> B*C*N
            label = (
                torch.Tensor(
                    [[int(self.label_to_cat_idx[label]) for label in data[0, -1, :]]]
                )
                .transpose(1, 0)
                .cuda()
            )  # 1*B -> B*1
            target = torch.from_numpy(
                np.squeeze(data[:, -1, :].astype(np.int32).transpose())
            ).cuda()  # N*C*N -> B*N

            vote_pool = torch.zeros(
                target.size()[0], target.size()[1], self.n_parts
            ).cuda()
            for _ in range(self.config["num_votes"]):
                # inference model ran here
                seg_pred, _ = classifier(
                    points, PointnetInference.to_categorical(label, self.n_classes)
                )
                vote_pool += seg_pred
            seg_pred = vote_pool / self.config["num_votes"]
            seg_pred = seg_pred.cpu().data.numpy()  # np[B * N * n_parts]
            pred_labels = np.argmax(seg_pred, axis=2)  # np[B*N]

            return pred_labels.T  # np[N*B]


if __name__ == "__main__":

    # default categories/classes
    segmentation_classes = {
        # fmt: off
        "Earphone"  : [16, 17, 18],
        "Motorbike" : [30, 31, 32, 33, 34, 35],
        "Rocket"    : [41, 42, 43],
        "Car"       : [8, 9, 10, 11],
        "Laptop"    : [28, 29],
        "Cap"       : [6, 7],
        "Skateboard": [44, 45, 46],
        "Mug"       : [36, 37],
        "Guitar"    : [19, 20, 21],
        "Bag"       : [4, 5],
        "Lamp"      : [24, 25, 26, 27],
        "Table"     : [47, 48, 49],
        "Airplane"  : [0, 1, 2, 3],
        "Pistol"    : [38, 39, 40],
        "Chair"     : [12, 13, 14, 15],
        "Knife"     : [22, 23],
        # fmt: on
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="batch size. Watch VRAM usage. [default: 24]",
    )
    parser.add_argument(
        "--gpu", type=str, default="0", help="specify gpu device [default: 0]"
    )
    parser.add_argument(
        "--num_point", type=int, default=2048, help="point Number [default: 2048]"
    )
    parser.add_argument(
        "--normal", action="store_true", default=False, help="use normals"
    )
    parser.add_argument(
        "--num_votes",
        type=int,
        default=3,
        help="aggregate segmentation scores with voting [default: 3]",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="pointnet2_part_seg_msg",
        help="Model to use [default: pointnet2_part_seg_msg]",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="experiment path. [default: log/part_seg/<model_name>/]",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
        help="data directory [default: data/shapenetcore_partanno_segmentation_benchmark_v0_normal]",
    )
    parser.add_argument(
        "--pt_path",
        type=str,
        default="",
        help="Pytorch file to use. [default: <log_dir>/checkpoints/best_model.pth]",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="",
        help="Path to save prediction output. Passing 'n/a' will disable output. [default: <log_dir>/inference_output]",
    )
    args = vars(parser.parse_args())
    args["gpu"] = int(args["gpu"])
    if args["log_dir"] == "":
        args["log_dir"] = "log/part_seg/" + args["model_name"]
    args["log_dir"] = Path(args["log_dir"]).resolve()
    args["data_dir"] = Path(args["data_dir"]).resolve()
    if args["pt_path"] == "":
        args["pt_path"] = args["log_dir"] / "checkpoints" / "best_model.pth"
    write_txt = True
    if args["out_path"] == "n/a":
        write_txt = False
    if args["out_path"] == "":
        args["out_path"] = args["log_dir"] / "inference_output"

    # read test file list
    with open(
        args["data_dir"] / "train_test_split" / "shuffled_test_file_list.json",
        "r",
        encoding="utf-8",
    ) as f:
        test_ids = tuple(set([str(d) for d in json.load(f)]))

    # instantiate class
    pointnet_seg = PointnetInference(
        config=args, seg_classes=segmentation_classes, log_events=True
    )
    pointnet_seg.log_string(
        "The number of data is: %d" % len(test_ids), pointnet_seg.log_events
    )

    # loop through txt files
    for i in tqdm(range(0, len(test_ids), args["batch_size"])):
        cur_bat_end = min(i + args["batch_size"], len(test_ids))
        txt_list = [
            args["data_dir"] / test_id.split("/")[-2] / f"{test_id.split('/')[-1]}.txt"
            for test_id in test_ids[i:cur_bat_end]
        ]

        # Run model
        pointnet_seg.log_string(
            f"Running model: {i} -> {cur_bat_end - 1}",
            pointnet_seg.log_events,
            False,
        )
        point_ary = txt_path_to_batch_tensor(
            txt_list, npoints=args["num_point"], normals=args["normal"], label=True
        )
        # prediction B*N
        prediction = pointnet_seg.main(data=point_ary)

        pred_data = point_ary.copy()
        pred_data[:, -1, :] = prediction

        # write results
        if not write_txt:
            continue
        for j in range(cur_bat_end - i):
            pointnet_seg.log_string(
                f"Writing data: {i} -> {i + cur_bat_end - 1}",
                pointnet_seg.log_events,
                False,
            )
            write_path = args["out_path"] / txt_list[j].parent.name / txt_list[j].name
            write_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(write_path, pred_data[:, :, j], delimiter=" ", header="")
