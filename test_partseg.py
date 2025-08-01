"""
Author: Benny
Date: Nov 2019
"""

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm, tqdm_notebook

from data_utils.ShapeNetDataLoader import PartNormalDataset

BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR
sys.path.append(str(ROOT_DIR / "models"))


class CommandLineArgs(argparse.Namespace):

    def __init__(self, **kwargs):
        """Create argument to pass into main() when colling from outside of the script such as Jupiter notebook.

        Args:
            **kwargs: arguments to pass into the main() function. **Dict can also be passed.
        """
        super().__init__(**kwargs)
        # define default arguments
        default_args = {
            # fmt: off
            "batch_size": 24,
            "gpu"       : "0", # str
            "num_point" : 2048,
            "log_root"  : "log",
            "log_dir"   : None,
            "normal"    : False,
            "num_votes" : 3,
            "data_dir"  : "data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
            'notebook'  : False
            # fmt: on
        }
        for key, value in default_args.items():
            if not self.__contains__(key):
                setattr(self, key, value)


def to_categorical(y, num_classes):
    """1-hot encodes a tensor"""
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser("PointNet")
    parser.add_argument(
        "--batch_size", type=int, default=24, help="batch size in testing [default: 24]"
    )
    parser.add_argument(
        "--gpu", type=str, default="0", help="specify gpu device [default: 0]"
    )
    parser.add_argument(
        "--num_point", type=int, default=2048, help="point Number [default: 2048]"
    )
    parser.add_argument(
        "--log_root", type=str, default="log", help="Log directory root [default: log]"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="experiment root within log directory",
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
        "--data_dir",
        type=str,
        default="data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
        help="data directory [default: data/shapenetcore_partanno_segmentation_benchmark_v0_normal]",
    )
    parser.add_argument(
        "--notebook",
        action="store_true",
        default=False,
        help="set if running from jupyter notebook.",
    )
    return parser.parse_args()


def main(args, seg_classes: dict):
    def log_string(log_str):
        logger.info(log_str)
        print(log_str)

    """HYPER PARAMETER"""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = Path(args.log_root) / "part_seg" / args.log_dir

    """LOG"""
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(experiment_dir / "eval.txt")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string("PARAMETER ...")
    log_string(args)

    data_dir = args.data_dir

    TEST_DATASET = PartNormalDataset(
        root=data_dir, npoints=args.num_point, split="test", normal_channel=args.normal
    )
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = len(seg_classes)  # 16
    num_part = len(
        [
            seg_id
            for seg_val_sublist in seg_classes.values()
            for seg_id in seg_val_sublist
        ]
    )  # 50

    """MODEL LOADING"""
    model_name = list((experiment_dir / "logs").iterdir())[0].stem
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    checkpoint = torch.load(
        experiment_dir / "checkpoints" / "best_model.pth", weights_only=False
    )
    classifier.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()
        if args.notebook:
            test_iter = tqdm_notebook(
                enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9
            )
        else:
            test_iter = tqdm(
                enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9
            )
        for _, (points, label, target) in test_iter:
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = (
                points.float().cuda(),
                label.long().cuda(),
                target.long().cuda(),
            )
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

            for _ in range(args.num_votes):
                seg_pred, _ = classifier(
                    points, to_categorical(label, num_classes)
                )  # model ran here
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = (
                    np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
                )

            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += cur_batch_size * NUM_POINT

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += np.sum((cur_pred_val == l) & (target == l))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                        np.sum(segp == l) == 0
                    ):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum(
                            (segl == l) & (segp == l)
                        ) / float(np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious:
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics["accuracy"] = total_correct / float(total_seen)
        test_metrics["class_avg_accuracy"] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64)
        )
        for cat in sorted(shape_ious.keys()):
            log_string(
                "eval mIoU of %s %f" % (cat + " " * (14 - len(cat)), shape_ious[cat])
            )
        test_metrics["class_avg_iou"] = mean_shape_ious
        test_metrics["instance_avg_iou"] = np.mean(all_shape_ious)

    log_string("Accuracy is: %.5f" % test_metrics["accuracy"])
    log_string("Class avg accuracy is: %.5f" % test_metrics["class_avg_accuracy"])
    log_string("Class avg mIOU is: %.5f" % test_metrics["class_avg_iou"])
    log_string("Instance avg mIOU is: %.5f" % test_metrics["instance_avg_iou"])

    return test_metrics, shape_ious, total_correct_class, total_seen_class


if __name__ == "__main__":

    # default categories/classes
    segmentation_classes = {
        "Earphone": [16, 17, 18],
        "Motorbike": [30, 31, 32, 33, 34, 35],
        "Rocket": [41, 42, 43],
        "Car": [8, 9, 10, 11],
        "Laptop": [28, 29],
        "Cap": [6, 7],
        "Skateboard": [44, 45, 46],
        "Mug": [36, 37],
        "Guitar": [19, 20, 21],
        "Bag": [4, 5],
        "Lamp": [24, 25, 26, 27],
        "Table": [47, 48, 49],
        "Airplane": [0, 1, 2, 3],
        "Pistol": [38, 39, 40],
        "Chair": [12, 13, 14, 15],
        "Knife": [22, 23],
    }

    arguments = parse_args()
    main(arguments, seg_classes=segmentation_classes)
