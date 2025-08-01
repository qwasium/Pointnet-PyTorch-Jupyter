"""
Author: Benny
Date: Nov 2019
"""

import argparse
import datetime
import importlib
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm, tqdm_notebook

import provider
from data_utils.ShapeNetDataLoader import PartNormalDataset

# import shutil


BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR
# sys.path.append(str(ROOT_DIR / 'models'))


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
            'model'        : 'pointnet_part_seg',
            'batch_size'   : 16,
            'epoch'        : 251,
            'learning_rate': 0.001,
            'gpu'          : '0',
            'optimizer'    : 'Adam',
            'log_dir'      : None,
            'log_root'     : 'log',
            'decay_rate'   : 1e-4,
            'npoint'       : 2048,
            'normal'       : False,
            'step_size'    : 20,
            'lr_decay'     : 0.5,
            'data_dir'     : 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal',
            'notebook'     : False
            # fmt: on
        }
        for key, value in default_args.items():
            if not self.__contains__(key):
                setattr(self, key, value)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """1-hot encodes a tensor"""
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser("Model")
    parser.add_argument(
        "--model",
        type=str,
        default="pointnet_part_seg",
        help="model name [default: pointnet_part_seg]",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch Size during training [default: 16]",
    )
    parser.add_argument(
        "--epoch", default=251, type=int, help="epoch to run [default: 251]"
    )
    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        help="initial learning rate [default: 0.001]",
    )
    parser.add_argument(
        "--gpu", type=str, default="0", help="specify GPU devices [default: 0]"
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="Adam or SGD [default: Adam]"
    )
    parser.add_argument(
        "--log_root", type=str, default="log", help="Log root directory [default: log]"
    )
    parser.add_argument(
        "--log_dir", type=str, default=None, help="log path wihin log root directory"
    )
    parser.add_argument(
        "--decay_rate", type=float, default=1e-4, help="weight decay [default: 1e-4]"
    )
    parser.add_argument(
        "--npoint", type=int, default=2048, help="point Number [default: 2048]"
    )
    parser.add_argument(
        "--normal", action="store_true", default=False, help="use normals"
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=20,
        help="decay step for lr decay [default: 20]",
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.5,
        help="decay rate for lr decay [default: 0.5]",
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

    """CREATE DIR"""
    timestr = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    exp_dir = Path(args.log_root)
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir / "part_seg"
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

    data_dir = args.data_dir

    # data_utils/ShapenetPartLoader.py
    TRAIN_DATASET = PartNormalDataset(
        root=data_dir, npoints=args.npoint, split="trainval", normal_channel=args.normal
    )
    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10,
        drop_last=True,
    )
    TEST_DATASET = PartNormalDataset(
        root=data_dir, npoints=args.npoint, split="test", normal_channel=args.normal
    )
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10
    )
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = 16
    num_part = 50

    """MODEL LOADING"""
    pwd = os.getcwd()
    os.chdir(ROOT_DIR)
    MODEL = importlib.import_module("models." + args.model)
    # shutil.copy('models/%s.py' % args.model, str(exp_dir))
    # shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    os.chdir(pwd)

    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("Linear") != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(checkpoints_dir / "best_model.pth")
        start_epoch = checkpoint["epoch"]
        classifier.load_state_dict(checkpoint["model_state_dict"])
        log_string("Use pretrain model")
    except:
        log_string("No existing model, starting training from scratch...")
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(), lr=args.learning_rate, momentum=0.9
        )

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_instance_avg_iou = 0

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string("Epoch %d (%d/%s):" % (global_epoch + 1, epoch + 1, args.epoch))

        """Adjust learning rate and BN momentum"""
        lr = max(
            args.learning_rate * (args.lr_decay ** (epoch // args.step_size)),
            LEARNING_RATE_CLIP,
        )
        log_string("Learning rate:%f" % lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        momentum = MOMENTUM_ORIGINAL * (
            MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP)
        )
        if momentum < 0.01:
            momentum = 0.01
        print("BN momentum updated to: %f" % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        """learning one epoch"""
        if args.notebook:
            train_iter = tqdm_notebook(
                enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9
            )
        else:
            train_iter = tqdm(
                enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9
            )
        for i, (points, label, target) in train_iter:
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = (
                points.float().cuda(),
                label.long().cuda(),
                target.long().cuda(),
            )
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(
                points, to_categorical(label, num_classes)
            )
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            loss = criterion(seg_pred, target, trans_feat)
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        log_string("Train accuracy is: %.5f" % train_instance_acc)

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            classifier = classifier.eval()

            if args.notebook:
                train_iter = tqdm_notebook(
                    enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9
                )
            else:
                train_iter = tqdm(
                    enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9
                )
            for _, (points, label, target) in train_iter:
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = (
                    points.float().cuda(),
                    label.long().cuda(),
                    target.long().cuda(),
                )
                points = points.transpose(2, 1)
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
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
                    total_correct_class[l] += np.sum(
                        (cur_pred_val == l) & (target == l)
                    )

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
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics["accuracy"] = total_correct / float(total_seen)
            test_metrics["class_avg_accuracy"] = np.mean(
                np.array(total_correct_class)
                / np.array(total_seen_class, dtype=np.float64)
            )
            for cat in sorted(shape_ious.keys()):
                log_string(
                    "eval mIoU of %s %f"
                    % (cat + " " * (14 - len(cat)), shape_ious[cat])
                )
            test_metrics["class_avg_iou"] = mean_shape_ious
            test_metrics["instance_avg_iou"] = np.mean(all_shape_ious)

        log_string(
            "Epoch %d test Accuracy: %f  Class avg mIOU: %f   Instance avg mIOU: %f"
            % (
                epoch + 1,
                test_metrics["accuracy"],
                test_metrics["class_avg_iou"],
                test_metrics["instance_avg_iou"],
            )
        )
        if test_metrics["instance_avg_iou"] >= best_instance_avg_iou:
            logger.info("Save model...")
            savepath = str(checkpoints_dir) + "/best_model.pth"
            log_string("Saving at %s" % savepath)
            state = {
                "epoch": epoch,
                "train_acc": train_instance_acc,
                "test_acc": test_metrics["accuracy"],
                "class_avg_iou": test_metrics["class_avg_iou"],
                "instance_avg_iou": test_metrics["instance_avg_iou"],
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string("Saving model....")

        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
        if test_metrics["class_avg_iou"] > best_class_avg_iou:
            best_class_avg_iou = test_metrics["class_avg_iou"]
        if test_metrics["instance_avg_iou"] > best_instance_avg_iou:
            best_instance_avg_iou = test_metrics["instance_avg_iou"]
        log_string("Best accuracy is: %.5f" % best_acc)
        log_string("Best class avg mIOU is: %.5f" % best_class_avg_iou)
        log_string("Best instance avg mIOU is: %.5f" % best_instance_avg_iou)
        global_epoch += 1

    return test_metrics


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
