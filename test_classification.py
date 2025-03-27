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
from tqdm import tqdm

from data_utils.ModelNetDataLoader import ModelNetDataLoader

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
            "use_cpu": False,
            "gpu": "0",
            "batch_size": 24,
            "num_category": 40,
            "num_point": 1024,
            "log_dir": None,
            "log_root": "log",
            "use_normals": False,
            "use_uniform_sample": False,
            "num_votes": 3,
            "data_dir": "data/modelnet40_normal_resampled",
        }
        for key, value in default_args.items():
            if not self.__contains__(key):
                setattr(self, key, value)


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser("Testing")
    parser.add_argument(
        "--use_cpu", action="store_true", default=False, help="use cpu mode"
    )
    parser.add_argument("--gpu", type=str, default="0", help="specify gpu device")
    parser.add_argument(
        "--batch_size", type=int, default=24, help="batch size in training"
    )
    parser.add_argument(
        "--num_category",
        default=40,
        type=int,
        choices=[10, 40],
        help="training on ModelNet10/40",
    )
    parser.add_argument("--num_point", type=int, default=1024, help="Point Number")
    parser.add_argument(
        "--log_root", type=str, default="log", help="Log directory root [default: log]"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Experiment root within log directory",
    )
    parser.add_argument(
        "--use_normals", action="store_true", default=False, help="use normals"
    )
    parser.add_argument(
        "--use_uniform_sample",
        action="store_true",
        default=False,
        help="use uniform sampiling",
    )
    parser.add_argument(
        "--num_votes",
        type=int,
        default=3,
        help="Aggregate classification scores with voting",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/modelnet40_normal_resampled",
        help="data directory [default: data/modelnet40_normal_resampled]",
    )
    return parser.parse_args()


def test(args, model, loader, num_class=40, vote_num=1):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    for _, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = (
                pred_choice[target == cat]
                .eq(target[target == cat].long().data)
                .cpu()
                .sum()
            )
            class_acc[cat, 0] += classacc.item() / float(
                points[target == cat].size()[0]
            )
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(log_str):
        logger.info(log_str)
        print(log_str)

    """HYPER PARAMETER"""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    """CREATE DIR"""
    experiment_dir = Path(args.log_root) / "classification" / args.log_dir

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

    """DATA LOADING"""
    log_string("Load dataset ...")
    data_path = args.data_dir

    test_dataset = ModelNetDataLoader(
        root=data_path, args=args, split="test", process_data=False
    )
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10
    )

    """MODEL LOADING"""
    num_class = args.num_category
    model_name = list((experiment_dir / "logs").iterdir())[0].stem
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(
        experiment_dir / "checkpoints" / "best_model.pth", weights_only=False
    )
    classifier.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        instance_acc, class_acc = test(
            args,
            classifier.eval(),
            testDataLoader,
            vote_num=args.num_votes,
            num_class=num_class,
        )
        log_string(
            "Test Instance Accuracy: %f, Class Accuracy: %f" % (instance_acc, class_acc)
        )


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
