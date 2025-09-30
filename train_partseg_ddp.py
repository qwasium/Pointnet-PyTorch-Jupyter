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
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm_notebook

import provider
from data_utils.ShapeNetDataLoader import PartNormalDataset


ROOT_DIR = Path(__file__).parent

class AddImportPath:
    """Temporarily add path to sys.path

    Usage
    -----
    with AddImportPath([<module directory>,...]):
        # import stuff
    """

    def __init__(self, mod_dir: list[os.PathLike]):
        mod_dir = [Path(fold).resolve() for fold in mod_dir]
        for fold in mod_dir:
            assert fold.exists(), f"Doesn't exist: {fold}"
        self.mod_dir = [str(fold) for fold in mod_dir]

    def __enter__(self):
        for fold in self.mod_dir:
            sys.path.insert(0, fold)

    def __exit__(self, exc_type, exc_value, traceback):
        for fold in self.mod_dir:
            try:
                sys.path.remove(fold)
            except Exception:
                warnings.warn(f"Failed to remove from sys.path: {fold}")

class TrainPartSegDDP:
    """Distributed training and evaluation for ShapeNet part segmentation.

    This class orchestrates dataset loading, DDP setup, training, validation,
    metric aggregation across ranks, and checkpointing.

    Config keys (dict)
    ------------------
    model: str
        Model module name under `models` providing `get_model` and `get_loss`.
    batch_size: int
    epoch: int
    learning_rate: float
    world_size: int
        Number of processes (ranks).
    optimizer: str
        "Adam" or "SGD".
    log_root: str
    log_dir: Optional[str]
    decay_rate: float
        Weight decay.
    npoint: int
        Number of points per sample.
    normal: bool
        Whether to use normals.
    step_size: int
        LR decay step.
    lr_decay: float
        LR decay rate.
    learning_rate_clip: float
        Minimum LR.
    bn_momentum: float
        Initial BN momentum.
    bn_momentum_decay: float
        BN momentum decay factor.
    data_dir: str
        Dataset root directory.
    notebook: bool
        TQDM in notebook on rank 0.
    master_addr: str
        Rendezvous address for process group.
    master_port: Union[str,int]
        Rendezvous port for process group.
    """

    def __init__(self, config: dict, seg_classes: dict[str, list[int]]):
        """Initialize configuration, logging, datasets, and model import path.

        Parameters
        ----------
        config : dict
            Training configuration parsed from CLI.
        seg_classes : dict[str, list[int]]
            Mapping from category name to list of part indices.
        """

        self.conf = config
        self.seg_classes = seg_classes
        self.n_classes = len(self.seg_classes)  # 16 categories
        self.n_parts = len(
            [
                seg_id
                for seg_val_sublist in self.seg_classes.values()
                for seg_id in seg_val_sublist
            ]
        )  # 50 part labels

        # mkdir
        timestr = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
        exp_dir = Path(self.conf["log_root"]) / "part_seg"
        exp_dir.mkdir(parents=True, exist_ok=True)
        if self.conf.get("log_dir") is None:
            exp_dir = exp_dir / timestr
        else:
            exp_dir = exp_dir / self.conf["log_dir"]
        exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = exp_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.log_dir = exp_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # logger
        self.logger = self._init_logger()
        self.log_string("PARAMETER ...")
        self.log_string(str(self.conf))

        # data_utils/ShapenetPartLoader.py
        self.train_dataset = PartNormalDataset(
            root=self.conf["data_dir"],
            npoints=self.conf["npoint"],
            split="trainval",
            normal_channel=self.conf["normal"],
        )
        self.test_dataset = PartNormalDataset(
            root=self.conf["data_dir"],
            npoints=self.conf["npoint"],
            split="test",
            normal_channel=self.conf["normal"],
        )
        self.log_string("The number of training data is: %d" % len(self.train_dataset))
        self.log_string("The number of test data is: %d" % len(self.test_dataset))

        # load model
        with AddImportPath([ROOT_DIR, ROOT_DIR / "models"]):
            try:
                self.model_module = importlib.import_module(self.conf["model"])
            except ImportError as err:
                self.log_string(
                    f"Failed to import model module '{self.conf['model']}' from cwd: {os.getcwd()}"
                )
                raise err

        # checkpoint metadata (actual weights loaded per-rank in _run_process)
        self.start_epoch = 0
        self.pretrained_path = None
        best_model = self.checkpoints_dir / "best_model.pth"
        if best_model.exists():
            self.pretrained_path = str(best_model)


    @staticmethod
    def inplace_relu(m):
        """Enable in-place behavior for ReLU modules to save memory."""
        classname = m.__class__.__name__
        if classname.find("ReLU") != -1:
            m.inplace = True

    @staticmethod
    def to_categorical(y, num_classes):
        """Return one-hot encoded labels with shape (N, num_classes)."""
        new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
        if y.is_cuda:
            return new_y.cuda()
        return new_y

    def _init_logger(self):
        """Create a file logger under the experiment log directory."""
        logger = logging.getLogger("Model")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler = logging.FileHandler(self.log_dir / f"{self.conf['model']}.txt")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def log_string(self, log_str, rank: int = 0):
        """Log and print a message only on rank 0."""
        if rank == 0:
            self.logger.info(log_str)
            print(log_str)

    def init_weights(self, m):
        """Initialize Conv/Linear layers with Xavier and zero biases."""
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("Linear") != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    @staticmethod
    def bn_momentum_adjust(m, momentum):
        """Set BatchNorm momentum on 1D/2D BN layers."""
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum


    def _run_process(self, rank):
        """Entry point for a single process in DDP training."""

        learning_rate_clip = self.conf.get("learning_rate_clip", 1e-5)
        bn_momentum_original = self.conf.get("bn_momentum", 0.1)
        bn_momentum_decay = self.conf.get("bn_momentum_decay", 0.5)
        bn_momentum_decay_step = self.conf["step_size"]

        device = self._setup_dist(rank)
        model, criterion = self._build_model(device)
        start_epoch = self._load_checkpoint_if_any(model, device, rank)
        optimizer = self._build_optimizer(model)
        ddp_model = DDP(model, device_ids=[rank] if device.type == "cuda" else None)
        train_dataloader, test_dataloader, train_sampler = self._build_dataloaders(rank)

        best_acc = 0.0
        best_class_avg_iou = 0.0
        best_instance_avg_iou = 0.0

        for epoch in range(start_epoch, self.conf["epoch"]):
            train_sampler.set_epoch(epoch)

            if rank == 0:
                self.log_string("Epoch %d/%d" % (epoch + 1, self.conf["epoch"]))

            self._adjust_schedules(epoch, optimizer, ddp_model, learning_rate_clip, bn_momentum_original, bn_momentum_decay, bn_momentum_decay_step)

            train_instance_acc = self._train_one_epoch(rank, device, ddp_model, optimizer, train_dataloader)

            test_metrics = self._validate_one_epoch(rank, device, ddp_model, test_dataloader)

            if rank == 0 and test_metrics is not None:
                self.log_string(
                    "Epoch %d test Accuracy: %f  Class avg mIOU: %f   Instance avg mIOU: %f"
                    % (
                        epoch + 1,
                        test_metrics["accuracy"],
                        test_metrics["class_avg_iou"],
                        test_metrics["instance_avg_iou"],
                    )
                )
                best_acc, best_class_avg_iou, best_instance_avg_iou = self._save_if_best(
                    epoch, ddp_model, optimizer, train_instance_acc, test_metrics, best_acc, best_class_avg_iou, best_instance_avg_iou
                )

        dist.barrier()
        dist.destroy_process_group()

        return None

    def _setup_dist(self, rank):
        """Initialize process group and return device for this rank."""
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
            backend = "nccl"
        else:
            backend = "gloo"
        # Set rendezvous variables from config (no hardcoded defaults here)
        master_addr = str(self.conf.get("master_addr", "127.0.0.1"))
        master_port = str(self.conf.get("master_port", "29500"))
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        dist.init_process_group(backend=backend, rank=rank, world_size=self.conf["world_size"])
        return device

    def _build_model(self, device):
        """Construct the model and loss function on the given device."""
        model = self.model_module.get_model(self.n_parts, normal_channel=self.conf["normal"]).to(device)
        criterion = self.model_module.get_loss()
        model.apply(TrainPartSegDDP.inplace_relu)
        return model, criterion

    def _load_checkpoint_if_any(self, model, device, rank):
        """Load a checkpoint if available and return the starting epoch."""
        start_epoch = 0
        if self.pretrained_path is not None and Path(self.pretrained_path).exists():
            try:
                checkpoint = torch.load(self.pretrained_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                start_epoch = int(checkpoint.get("epoch", 0))
                if rank == 0:
                    self.log_string("Use pretrain model from %s" % self.pretrained_path, rank=rank)
            except Exception as err:
                if rank == 0:
                    self.log_string(f"Failed to load checkpoint: {err}", rank=rank)
        return start_epoch

    def _build_optimizer(self, model):
        """Create the optimizer from config for the provided model."""
        if self.conf["optimizer"].lower() == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.conf["learning_rate"],
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.conf["decay_rate"],
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.conf["learning_rate"], momentum=0.9
            )
        return optimizer

    def _build_dataloaders(self, rank):
        """Create distributed samplers and data loaders for train and test."""
        train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.conf["world_size"], rank=rank, shuffle=True)
        test_sampler = DistributedSampler(self.test_dataset, num_replicas=self.conf["world_size"], rank=rank, shuffle=False)
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.conf["batch_size"],
            shuffle=False,
            num_workers=10,
            drop_last=True,
            sampler=train_sampler,
        )
        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.conf["batch_size"],
            shuffle=False,
            num_workers=10,
            sampler=test_sampler,
        )
        return train_dataloader, test_dataloader, train_sampler

    def _adjust_schedules(self, epoch, optimizer, ddp_model, learning_rate_clip, bn_momentum_original, bn_momentum_decay, bn_momentum_decay_step):
        """Adjust learning rate and BatchNorm momentum for the current epoch."""
        lr = max(
            self.conf["learning_rate"] * (self.conf["lr_decay"] ** (epoch // self.conf["step_size"])),
            learning_rate_clip,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        momentum = bn_momentum_original * (bn_momentum_decay ** (epoch // bn_momentum_decay_step))
        if momentum < 0.01:
            momentum = 0.01
        def _set_bn_momentum(m):
            return TrainPartSegDDP.bn_momentum_adjust(m, momentum)
        ddp_model.module.apply(_set_bn_momentum)
        ddp_model.train()

    def _train_one_epoch(self, rank, device, ddp_model, optimizer, train_dataloader):
        """Run one training epoch and return globally averaged train accuracy."""
        mean_correct = []
        if self.conf.get("notebook", False) and rank == 0:
            train_iter = tqdm_notebook(enumerate(train_dataloader), total=len(train_dataloader), smoothing=0.9)
        else:
            train_iter = enumerate(train_dataloader)
        for _, (points, label, target) in train_iter:
            optimizer.zero_grad(set_to_none=True)
            points_np = points.numpy()
            points_np[:, :, 0:3] = provider.random_scale_point_cloud(points_np[:, :, 0:3])
            points_np[:, :, 0:3] = provider.shift_point_cloud(points_np[:, :, 0:3])
            points = torch.from_numpy(points_np)
            points = points.float().to(device)
            label = label.long().to(device)
            target = target.long().to(device)
            points = points.transpose(2, 1)
            seg_pred, trans_feat = ddp_model(points, TrainPartSegDDP.to_categorical(label, self.n_classes))
            seg_pred = seg_pred.contiguous().view(-1, self.n_parts)
            target_flat = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target_flat.data).cpu().sum()
            mean_correct.append(correct.item() / (self.conf["batch_size"] * self.conf["npoint"]))
            loss = self.model_module.get_loss()(seg_pred, target_flat, trans_feat)
            loss.backward()
            optimizer.step()
        train_instance_acc = float(np.mean(mean_correct)) if len(mean_correct) > 0 else 0.0
        train_acc_tensor = torch.tensor(train_instance_acc, device=device)
        dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.SUM)
        train_instance_acc = (train_acc_tensor.item() / self.conf["world_size"]) if self.conf["world_size"] > 0 else train_instance_acc
        if rank == 0:
            self.log_string("Train accuracy is: %.5f" % train_instance_acc)
        return train_instance_acc

    def _validate_one_epoch(self, rank, device, ddp_model, test_dataloader):
        """Validate over the test set and return reduced metrics on rank 0."""
        with torch.no_grad():
            ddp_model.eval()
            total_correct = torch.tensor(0.0, device=device)
            total_seen = torch.tensor(0.0, device=device)
            total_seen_class = torch.zeros(self.n_parts, device=device)
            total_correct_class = torch.zeros(self.n_parts, device=device)
            cats_sorted = sorted(self.seg_classes.keys())
            num_cats = len(cats_sorted)
            cat_to_idx = {cat: idx for idx, cat in enumerate(cats_sorted)}
            cat_iou_sum = torch.zeros(num_cats, device=device)
            cat_iou_cnt = torch.zeros(num_cats, device=device)
            instance_iou_sum = torch.tensor(0.0, device=device)
            instance_cnt = torch.tensor(0.0, device=device)
            seg_label_to_cat = {}
            for cat in self.seg_classes.keys():
                for lab in self.seg_classes[cat]:
                    seg_label_to_cat[lab] = cat
            if self.conf.get("notebook", False) and rank == 0:
                test_iter = tqdm_notebook(enumerate(test_dataloader), total=len(test_dataloader), smoothing=0.9)
            else:
                test_iter = enumerate(test_dataloader)
            for _, (points, label, target) in test_iter:
                cur_batch_size, NUM_POINT, _ = points.size()
                points = points.float().to(device)
                label = label.long().to(device)
                target = target.long().to(device)
                points = points.transpose(2, 1)
                seg_pred, _ = ddp_model(points, TrainPartSegDDP.to_categorical(label, self.n_classes))
                cur_pred_val_logits = seg_pred.detach().cpu().numpy()
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target_np = target.detach().cpu().numpy()
                for bi in range(cur_batch_size):
                    cat = seg_label_to_cat[target_np[bi, 0]]
                    logits = cur_pred_val_logits[bi, :, :]
                    cur_pred_val[bi, :] = (
                        np.argmax(logits[:, self.seg_classes[cat]], 1) + self.seg_classes[cat][0]
                    )
                correct = np.sum(cur_pred_val == target_np)
                total_correct += torch.tensor(float(correct), device=device)
                total_seen += torch.tensor(float(cur_batch_size * NUM_POINT), device=device)
                for l in range(self.n_parts):
                    total_seen_class[l] += torch.tensor(float(np.sum(target_np == l)), device=device)
                    total_correct_class[l] += torch.tensor(float(np.sum((cur_pred_val == l) & (target_np == l))), device=device)
                for bi in range(cur_batch_size):
                    segp = cur_pred_val[bi, :]
                    segl = target_np[bi, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(self.seg_classes[cat]))]
                    for l in self.seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                            part_ious[l - self.seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - self.seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l))
                            )
                    shape_iou = float(np.mean(part_ious))
                    cat_iou_sum[cat_to_idx[cat]] += shape_iou
                    cat_iou_cnt[cat_to_idx[cat]] += 1.0
                    instance_iou_sum += shape_iou
                    instance_cnt += 1.0
            dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_seen, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_correct_class, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_seen_class, op=dist.ReduceOp.SUM)
            dist.all_reduce(cat_iou_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(cat_iou_cnt, op=dist.ReduceOp.SUM)
            dist.all_reduce(instance_iou_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(instance_cnt, op=dist.ReduceOp.SUM)
            if rank == 0:
                test_metrics = {}
                test_metrics["accuracy"] = (total_correct.item() / float(total_seen.item())) if total_seen.item() > 0 else 0.0
                seen_class_np = total_seen_class.cpu().numpy()
                correct_class_np = total_correct_class.cpu().numpy()
                with np.errstate(divide='ignore', invalid='ignore'):
                    class_avg_acc = np.nan_to_num(correct_class_np / seen_class_np)
                test_metrics["class_avg_accuracy"] = float(np.mean(class_avg_acc))
                cat_iou_sum_np = cat_iou_sum.cpu().numpy()
                cat_iou_cnt_np = cat_iou_cnt.cpu().numpy()
                with np.errstate(divide='ignore', invalid='ignore'):
                    per_cat_miou = np.nan_to_num(cat_iou_sum_np / np.maximum(cat_iou_cnt_np, 1e-12))
                for idx, cat in enumerate(cats_sorted):
                    self.log_string(
                        "eval mIoU of %s %f" % (cat + " " * (14 - len(cat)), per_cat_miou[idx]),
                        rank=rank,
                    )
                test_metrics["class_avg_iou"] = float(np.mean(per_cat_miou)) if len(per_cat_miou) > 0 else 0.0
                test_metrics["instance_avg_iou"] = (
                    float(instance_iou_sum.item() / max(instance_cnt.item(), 1.0))
                    if instance_cnt.item() > 0 else 0.0
                )
            else:
                test_metrics = None
            return test_metrics

    def _save_if_best(self, epoch, ddp_model, optimizer, train_instance_acc, test_metrics, best_acc, best_class_avg_iou, best_instance_avg_iou):
        """Save checkpoint on rank 0 if metrics improve and return updated bests."""
        if test_metrics["instance_avg_iou"] >= best_instance_avg_iou:
            self.logger.info("Save model...")
            savepath = str(self.checkpoints_dir / "best_model.pth")
            self.log_string("Saving at %s" % savepath)
            state = {
                "epoch": epoch,
                "train_acc": train_instance_acc,
                "test_acc": test_metrics["accuracy"],
                "class_avg_iou": test_metrics["class_avg_iou"],
                "instance_avg_iou": test_metrics["instance_avg_iou"],
                "model_state_dict": ddp_model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(state, savepath)
            self.log_string("Saving model....")
        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
        if test_metrics["class_avg_iou"] > best_class_avg_iou:
            best_class_avg_iou = test_metrics["class_avg_iou"]
        if test_metrics["instance_avg_iou"] > best_instance_avg_iou:
            best_instance_avg_iou = test_metrics["instance_avg_iou"]
        self.log_string("Best accuracy is: %.5f" % best_acc)
        self.log_string("Best class avg mIOU is: %.5f" % best_class_avg_iou)
        self.log_string("Best instance avg mIOU is: %.5f" % best_instance_avg_iou)
        return best_acc, best_class_avg_iou, best_instance_avg_iou

    @staticmethod
    def _mp_entry(rank: int, conf: dict, seg_classes: dict[str, list[int]]):
        """Spawn entry point. Construct trainer in each process and run worker."""
        trainer = TrainPartSegDDP(conf, seg_classes)
        trainer._run_process(rank)

    def main(self):
        results = mp.spawn(
            # Avoid pickling the class instance by spawning a static entry point
            TrainPartSegDDP._mp_entry,
            args=(self.conf, self.seg_classes),
            nprocs=self.conf["world_size"], join=True
        )
        return results

if __name__ == "__main__":

    # fmt: off
    # default categories/classes
    segmentation_classes = {
        "Earphone"  : [16, 17, 18],
        "Motorbike" : [30, 31, 32, 33, 34, 35],
        "Rocket"    : [41, 42, 43],
        "Car"       : [8,  9,  10, 11],
        "Laptop"    : [28, 29],
        "Cap"       : [6,  7],
        "Skateboard": [44, 45, 46],
        "Mug"       : [36, 37],
        "Guitar"    : [19, 20, 21],
        "Bag"       : [4,  5],
        "Lamp"      : [24, 25, 26, 27],
        "Table"     : [47, 48, 49],
        "Airplane"  : [0,  1,  2,  3],
        "Pistol"    : [38, 39, 40],
        "Chair"     : [12, 13, 14, 15],
        "Knife"     : [22, 23],
    }

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
        "--epoch",
        default=251,
        type=int,
        help="epoch to run [default: 251]"
    )
    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        help="initial learning rate [default: 0.001]",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=8,
        help="Pytorch DDP world size."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="Adam or SGD [default: Adam]"
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="log",
        help="Log root directory [default: log]"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="log path wihin log root directory"
    )
    parser.add_argument(
        "--decay_rate",
        type=float,
        default=1e-4,
        help="weight decay [default: 1e-4]"
    )
    parser.add_argument(
        "--npoint",
        type=int,
        default=2048,
        help="point Number [default: 2048]"
    )
    parser.add_argument(
        "--normal",
        action="store_true",
        default=False,
        help="use normals"
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
        "--learning_rate_clip",
        type=float,
        default=1e-5,
        help="minimum learning rate clip [default: 1e-5]",
    )
    parser.add_argument(
        "--bn_momentum",
        type=float,
        default=0.1,
        help="initial BatchNorm momentum [default: 0.1]",
    )
    parser.add_argument(
        "--bn_momentum_decay",
        type=float,
        default=0.5,
        help="BatchNorm momentum decay factor [default: 0.5]",
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
    parser.add_argument(
        "--master_addr",
        type=str,
        default="127.0.0.1",
        help="rendezvous address for init_process_group",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default="29500",
        help="rendezvous port for init_process_group",
    )
    # fmt: on

    arguments = vars(parser.parse_args())
    trainer = TrainPartSegDDP(arguments, segmentation_classes)
    trainer.main()