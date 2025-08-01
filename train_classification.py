"""
Author: Benny
Date: Nov 2019
"""

import argparse
import datetime
import importlib
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm, tqdm_notebook

import provider
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR
sys.path.append(str(ROOT_DIR / "models"))


class CommandLineArgs(argparse.Namespace):

    def __init__(self, **kwargs):
        """Create argument to pass into main() when colling from outside of the script such as Jupiter Notebook.

        Args:
            **kwargs: arguments to pass into the main() function. **Dict can also be passed.
        """
        super().__init__(**kwargs)
        # define default arguments
        default_args = {
            # fmt: off
            'use_cpu'           : False,
            'gpu'               : '0', # str
            'batch_size'        : 25,
            'model'             : 'pointnet_cls',
            'num_category'      : 40,
            'epoch'             : 200,
            'learning_rate'     : 0.001,
            'num_point'         : 1024,
            'optimizer'         : 'adam',
            'log_root'          : 'log',
            'log_dir'           : None,
            'decay_rate'        : 1e-4,
            'use_normals'       : False,
            'process_data'      : False,
            'use_uniform_sample': False,
            'data_dir'          : 'data/modelnet40_normal_resampled',
            'notebook'          : False
            # fmt: on
        }
        for key, value in default_args.items():
            if not self.__contains__(key):
                setattr(self, key, value)


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser("training")
    parser.add_argument(
        "--use_cpu", action="store_true", default=False, help="use cpu mode"
    )
    parser.add_argument("--gpu", type=str, default="0", help="specify gpu device")
    parser.add_argument(
        "--batch_size", type=int, default=24, help="batch size in training"
    )
    parser.add_argument(
        "--model", default="pointnet_cls", help="model name [default: pointnet_cls]"
    )
    parser.add_argument(
        "--num_category",
        default=40,
        type=int,
        choices=[10, 40],
        help="training on ModelNet10/40",
    )
    parser.add_argument(
        "--epoch", default=200, type=int, help="number of epoch in training"
    )
    parser.add_argument(
        "--learning_rate", default=0.001, type=float, help="learning rate in training"
    )
    parser.add_argument("--num_point", type=int, default=1024, help="Point Number")
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="optimizer for training"
    )
    parser.add_argument(
        "--log_root", type=str, default="log", help="log directory root [default: log]"
    )
    parser.add_argument(
        "--log_dir", type=str, default=None, help="experiment root within log directory"
    )
    parser.add_argument("--decay_rate", type=float, default=1e-4, help="decay rate")
    parser.add_argument(
        "--use_normals", action="store_true", default=False, help="use normals"
    )
    parser.add_argument(
        "--process_data", action="store_true", default=False, help="save data offline"
    )
    parser.add_argument(
        "--use_uniform_sample",
        action="store_true",
        default=False,
        help="use uniform sampiling",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/modelnet40_normal_resampled",
        help="data directory [default: data/modelnet40_normal_resampled]",
    )
    parser.add_argument(
        "--notebook",
        action="store_true",
        default=False,
        help="set if running from jupyter notebook.",
    )
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


def test(args, model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    if args.notebook:
        test_iter = tqdm_notebook(enumerate(loader), total=len(loader))
    else:
        test_iter = tqdm(enumerate(loader), total=len(loader))
    for _, (points, target) in test_iter:

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
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
    timestr = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    exp_dir = Path(args.log_root)
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir / "classification"
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir / timestr
    else:
        exp_dir = exp_dir / args.log_dir
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    """LOG"""
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(log_dir / f"{args.model}.txt")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string("PARAMETER ...")
    log_string(args)

    """DATA LOADING"""
    log_string("Load dataset ...")
    data_path = args.data_dir

    train_dataset = ModelNetDataLoader(
        root=data_path, args=args, split="train", process_data=args.process_data
    )
    test_dataset = ModelNetDataLoader(
        root=data_path, args=args, split="test", process_data=args.process_data
    )
    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10,
        drop_last=True,
    )
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10
    )

    """MODEL LOADING"""
    num_class = args.num_category
    model = importlib.import_module(args.model)
    # shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    # shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    # shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(checkpoints_dir / "best_model.pth")
        start_epoch = checkpoint["epoch"]
        classifier.load_state_dict(checkpoint["model_state_dict"])
        log_string("Use pretrain model")
    except:
        log_string("No existing model, starting training from scratch...")
        start_epoch = 0

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    """TRANING"""
    logger.info("Start training...")
    for epoch in range(start_epoch, args.epoch):
        log_string("Epoch %d (%d/%s):" % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        if args.notebook:
            train_iter = tqdm_notebook(
                enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9
            )
        else:
            train_iter = tqdm(
                enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9
            )
        for _batch_id, (points, target) in train_iter:
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string("Train Instance Accuracy: %f" % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(
                args, classifier.eval(), testDataLoader, num_class=num_class
            )

            if instance_acc >= best_instance_acc:
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if class_acc >= best_class_acc:
                best_class_acc = class_acc
            log_string(
                "Test Instance Accuracy: %f, Class Accuracy: %f"
                % (instance_acc, class_acc)
            )
            log_string(
                "Best Instance Accuracy: %f, Class Accuracy: %f"
                % (best_instance_acc, best_class_acc)
            )

            if instance_acc >= best_instance_acc:
                logger.info("Save model...")
                savepath = str(checkpoints_dir) + "/best_model.pth"
                log_string("Saving at %s" % savepath)
                state = {
                    "epoch": best_epoch,
                    "instance_acc": instance_acc,
                    "class_acc": class_acc,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info("End of training...")


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
