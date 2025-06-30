"""See rot_shr_flip_dichot.ipynb"""

import os
from pathlib import Path
from pprint import pprint
from argparse import ArgumentParser
import shutil

from exp_utils import AddPath, PointnetPath

HERE = Path(__file__).resolve().parent
with AddPath(HERE.parent):
    import train_partseg as train
    import test_partseg as test


def main(
    gpu_idx: int | str,
    path_obj: PointnetPath,
    do_train: bool = True,
    do_test: bool = True,
    seg_classes: dict | None = None,
    train_args: dict | None = None,
    test_args: dict | None = None,
) -> (dict, float, dict):
    gpu_idx = str(gpu_idx)

    # fmt: off
    if seg_classes is None:
        seg_classes = {
            "paprika": [0, 1],
            # 0: non-leaves
            # 1: leaves

            # padding for 2:49
            'earphone'  : [16, 17, 18],
            'motorbike' : [30, 31, 32, 33, 34, 35],
            'rocket'    : [41, 42, 43],
            'car'       : [8, 9, 10, 11],
            'laptop'    : [28, 29],
            'cap'       : [6, 7],
            'skateboard': [44, 45, 46],
            'mug'       : [36, 37],
            'guitar'    : [19, 20, 21],
            'bag'       : [2, 3, 4, 5],
            'lamp'      : [24, 25, 26, 27],
            'table'     : [47, 48, 49],
            'pistol'    : [38, 39, 40],
            'chair'     : [12, 13, 14, 15],
            'knife'     : [22, 23]
        }
    if train_args is None:
        train_args = {
            # model params
            "model"     : "pointnet2_part_seg_msg",
            # "model"    : "pointnet2_part_seg_ssg",

            # data params
            "normal"    : True,
            "log_root"  : path_obj.log_root,
            "log_dir"   : path_obj.log_dir,
            "data_dir"  : path_obj.data_dir,

            # training params
            "gpu"       : gpu_idx,
            # "npoint"    : 2048,
            # "batch_size": 16,
            # "decay_rate": 1e-4,
            # "step_size" : 20,
            # "lr_decay"  : 0.5,
            #"epoch"      : 500,
            "epoch": 10
            # "optimizer" : "adam",
        }
    if test_args is None:
        test_args = {
            # data params
            "normal"    : True,
            "log_root"  : path_obj.log_root,
            "log_dir"   : path_obj.log_dir,
            "data_dir"  : path_obj.data_dir,

            # testing params
            "gpu"       : gpu_idx,
            # "num_points": 2048,
            # "batch_size": 24,
            # "num_votes" : 3,
        }
    # fmt: on

    if do_train:
        train.main(train.CommandLineArgs(**train_args), seg_classes)

        # backup weights
        chk_pt_dir = path_obj.log_root / "part_seg" / path_obj.log_dir / "checkpoints"
        epochs = [
            int(f.stem.split("_")[-1])
            for f in chk_pt_dir.iterdir()
            if f.name != "best_model.pth" and f.name == ".pth"
        ]
        cp_pt = chk_pt_dir / f"epoch_{max(epochs) + train_args["epoch"]}.pth"
        assert not cp_pt.exists()
        shutils.copy2(src=chk_pt_dir / "best_model.pth", dst=cp_pt)

    if do_test:
        (test_metrics, shape_ious, total_correct_class, total_seen_class) = test.main(
            test.CommandLineArgs(**test_args), seg_classes
        )
        seg_ids = [
            seg_id
            for seg_val_sublist in seg_classes.values()
            for seg_id in seg_val_sublist
        ]
        seg_correct = dict(zip(range(len(seg_ids)), total_correct_class))
        seg_total = dict(zip(range(len(seg_ids)), total_seen_class))
        seg_acc = {}
        for id, correct_n in seg_correct.items():
            total_n = seg_total[id]
            if total_n == 0:
                seg_acc[id] = 0
            else:
                seg_acc[id] = correct_n / total_n
        seg_class_acc = {}
        for cat in seg_classes:
            seg_class_acc[cat] = {}
            for id in seg_classes[cat]:
                seg_class_acc[cat][id] = seg_acc[id]
        pprint(seg_class_acc["paprika"])
        return test_metrics, shape_ious["paprika"], seg_class_acc["paprika"]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu", "-g", help="Number of GPU to use.")
    parser.add_argument(
        "--experiment", "-e", default="rot-shr-flip-dichot", help="Experiment name."
    )
    parser.add_argument("--no-train", action="store_false", default=True, help="Skip training.")
    parser.add_argument("--no-test", action="store_false", default=True, help="Skip testing.")
    args = parser.parse_args()
    with AddPath(HERE.parent):
        met, iou, acc = main(
            gpu_idx=args.gpu,
            path_obj=PointnetPath(args.experiment),
            do_train=args.no_train,
            do_test=args.no_test
        )
    pprint(met)
    print(iou)
    pprint(acc)


