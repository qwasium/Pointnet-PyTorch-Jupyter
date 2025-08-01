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
import time
from pathlib import Path

import numpy as np
import torch

# import shutil
from tqdm import tqdm, tqdm_notebook

import provider
from data_utils.S3DISDataLoader import S3DISDataset

BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR
sys.path.append(str(ROOT_DIR / "models"))

classes = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
]
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


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
            "model"        : "pointnet_sem_seg",
            "batch_size"   : 16,
            "epoch"        : 32,
            "learning_rate": 0.001,
            "gpu"          : "0",
            "optimizer"    : "Adam",
            "log_dir"      : None,
            "log_root"     : "log",
            "decay_rate"   : 1e-4,
            "npoint"       : 4096,
            "step_size"    : 10,
            "lr_decay"     : 0.7,
            "test_area"    : 5,
            "data_dir"     : "data/stanford_indoor3d/",
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


def parse_args():
    parser = argparse.ArgumentParser("Model")
    parser.add_argument(
        "--model",
        type=str,
        default="pointnet_sem_seg",
        help="model name [default: pointnet_sem_seg]",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch Size during training [default: 16]",
    )
    parser.add_argument(
        "--epoch", default=32, type=int, help="Epoch to run [default: 32]"
    )
    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        help="Initial learning rate [default: 0.001]",
    )
    parser.add_argument(
        "--gpu", type=str, default="0", help="GPU to use [default: GPU 0]"
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="Adam or SGD [default: Adam]"
    )
    parser.add_argument(
        "--log_root", type=str, default="log", help="Log root directory [default: log]"
    )
    parser.add_argument(
        "--log_dir", type=str, default=None, help="Log path [default: None]"
    )
    parser.add_argument(
        "--decay_rate", type=float, default=1e-4, help="weight decay [default: 1e-4]"
    )
    parser.add_argument(
        "--npoint", type=int, default=4096, help="Point Number [default: 4096]"
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="Decay step for lr decay [default: every 10 epochs]",
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.7,
        help="Decay rate for lr decay [default: 0.7]",
    )
    parser.add_argument(
        "--test_area",
        type=int,
        default=5,
        help="Which area to use for test, option: 1-6 [default: 5]",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/stanford_indoor3d/",
        help="data directory [default: data/stanford_indoor3d/]",
    )
    parser.add_argument(
        "--notebook",
        action="store_true",
        default=False,
        help="set if running from jupyter notebook.",
    )
    return parser.parse_args()


def main(args):
    def log_string(log_str):
        logger.info(log_str)
        print(log_str)

    """HYPER PARAMETER"""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    """CREATE DIR"""
    timestr = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    experiment_dir = Path(args.log_root)
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir / "sem_seg"
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir / timestr
    else:
        experiment_dir = experiment_dir / args.log_dir
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir / "logs"
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
    NUM_CLASSES = 13
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = S3DISDataset(
        split="train",
        data_root=data_dir,
        num_point=NUM_POINT,
        test_area=args.test_area,
        block_size=1.0,
        sample_rate=1.0,
        transform=None,
        notebook=args.notebook,
    )
    print("start loading test data ...")
    TEST_DATASET = S3DISDataset(
        split="test",
        data_root=data_dir,
        num_point=NUM_POINT,
        test_area=args.test_area,
        block_size=1.0,
        sample_rate=1.0,
        transform=None,
        notebook=args.notebook,
    )

    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time())),
    )
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        drop_last=True,
    )
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    """MODEL LOADING"""
    MODEL = importlib.import_module(args.model)
    # shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    # shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).cuda()
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
        checkpoint = torch.load(Path(experiment_dir, "checkpoints/best_model.pth"))
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

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
        """Train on chopped scenes"""
        log_string(
            "**** Epoch %d (%d/%s) ****" % (global_epoch + 1, epoch + 1, args.epoch)
        )
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
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        if args.notebook:
            train_iter = tqdm_notebook(
                enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9
            )
        else:
            train_iter = tqdm(
                enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9
            )
        for _, (points, target) in train_iter:
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += BATCH_SIZE * NUM_POINT
            loss_sum += loss
        log_string("Training mean loss: %f" % (loss_sum / num_batches))
        log_string("Training accuracy: %f" % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            logger.info("Save model...")
            savepath = str(checkpoints_dir) + "/model.pth"
            log_string("Saving at %s" % savepath)
            state = {
                "epoch": epoch,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string("Saving model....")

        """Evaluate on chopped scenes"""
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log_string("---- EPOCH %03d EVALUATION ----" % (global_epoch + 1))
            if args.notebook:
                train_iter = tqdm_notebook(
                    enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9
                )
            else:
                train_iter = tqdm(
                    enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9
                )
            for _, (points, target) in train_iter:
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += BATCH_SIZE * NUM_POINT
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum(
                        (pred_val == l) & (batch_label == l)
                    )
                    total_iou_deno_class[l] += np.sum(
                        ((pred_val == l) | (batch_label == l))
                    )

            labelweights = labelweights.astype(float) / np.sum(
                labelweights.astype(float)
            )
            mIoU = np.mean(
                np.array(total_correct_class)
                / (np.array(total_iou_deno_class, dtype=float) + 1e-6)
            )
            log_string("eval mean loss: %f" % (loss_sum / float(num_batches)))
            log_string("eval point avg class IoU: %f" % (mIoU))
            log_string("eval point accuracy: %f" % (total_correct / float(total_seen)))
            log_string(
                "eval point avg class acc: %f"
                % (
                    np.mean(
                        np.array(total_correct_class)
                        / (np.array(total_seen_class, dtype=float) + 1e-6)
                    )
                )
            )

            iou_per_class_str = "------- IoU --------\n"
            for l in range(NUM_CLASSES):
                iou_per_class_str += "class %s weight: %.3f, IoU: %.3f \n" % (
                    seg_label_to_cat[l] + " " * (14 - len(seg_label_to_cat[l])),
                    labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]),
                )

            log_string(iou_per_class_str)
            log_string("Eval mean loss: %f" % (loss_sum / num_batches))
            log_string("Eval accuracy: %f" % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info("Save model...")
                savepath = Path(checkpoints_dir, "best_model.pth")
                log_string("Saving at %s" % savepath)
                state = {
                    "epoch": epoch,
                    "class_avg_iou": mIoU,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string("Saving model....")
            log_string("Best mIoU: %f" % best_iou)
        global_epoch += 1


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
