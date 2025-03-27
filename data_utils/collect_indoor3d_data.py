import argparse
import os
import sys
from pathlib import Path

from indoor3d_util import collect_point_label

BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(BASE_DIR))


def collect_indoor3d_data(
    data_path: str | Path, output_path: str | Path, skip_existing: bool = True
) -> None:
    """
    Collect and process indoor3d data from the specified data path.

    Args:
        data_path (str or Path): Path to the indoor3d data directory
            Standard path: data/s3dis/Stanford3dDataset_v1.2_Aligned_Version
        output_path (str or Path, optional): Path where processed data will be saved.
            If None, will save in a 'stanford_indoor3d' directory next to data_path.
            Standard path: data/stanford_indoor3d
        skip_existing (bool): To skip if output file already exists.
    Returns:
        Path: Path to the output directory
    """
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    anno_paths = [
        line.rstrip()
        for line in open(Path(BASE_DIR, "meta", "anno_paths.txt"), encoding="utf-8")
    ]
    anno_paths = [data_path / p for p in anno_paths]

    # Note:
    # There is an extra character in the v1.2 data in Area_5/hallway_6/Annotations/ceiling_1.txt.
    # It must be fixed manually.
    for anno_path in anno_paths:
        print(anno_path)
        elements = anno_path.parts
        out_filename = f"{elements[-3]}_{elements[-2]}.npy"  # Area_1_hallway_1.npy
        collect_point_label(
            anno_path, output_path / out_filename, "numpy", skip_existing
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default="data/s3dis/Stanford3dDataset_v1.2_Aligned_Version",
        help="Path to the indoor3d data directory. Default: data/s3dis/Stanford3dDataset_v1.2_Aligned_Version",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="data/stanford_indoor3d",
        help="Path where processed data will be saved. Default: data/stanford_indoor3d",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        required=False,
        help="If output file already exist, skip rather than overwriting.",
    )
    args = parser.parse_args()
    collect_indoor3d_data(args.data_path, args.output_path, args.skip_existing)
    print(f"Processed data saved to {args.output_path}")
