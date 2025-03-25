import sys
import argparse
from pathlib import Path
from indoor3d_util import collect_point_label

BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(BASE_DIR))

def collect_indoor3d_data(data_path: str | Path, output_path: str | Path):
    """
    Collect and process indoor3d data from the specified data path.

    Args:
        data_path (str or Path): Path to the indoor3d data directory
            Standard path: data/s3dis/Stanford3dDataset_v1.2_Aligned_Version
        output_path (str or Path, optional): Path where processed data will be saved.
            If None, will save in a 'stanford_indoor3d' directory next to data_path.
            Standard path: data/stanford_indoor3d
    Returns:
        Path: Path to the output directory
    """
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    anno_paths = [line.rstrip() for line in open(Path(BASE_DIR,'meta/anno_paths.txt'), encoding='utf-8')]
    anno_paths = [data_path / p for p in anno_paths]

    # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
    for anno_path in anno_paths:
        print(anno_path)
        try:
            elements = anno_path.parts
            out_filename = f"{elements[-3]}_{elements[-2]}.npy"  # Area_1_hallway_1.npy
            collect_point_label(str(anno_path), str(output_path / out_filename), 'numpy')
        except:
            print(anno_path, 'ERROR!!')

    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', required=False, default='data/s3dis/Stanford3dDataset_v1.2_Aligned_Version',
        help='Path to the indoor3d data directory. Default: data/s3dis/Stanford3dDataset_v1.2_Aligned_Version'
    )
    parser.add_argument(
        '--output_path', required=False, default='data/stanford_indoor3d',
        help='Path where processed data will be saved. Default: data/stanford_indoor3d'
    )
    args = parser.parse_args()
    output_path = collect_indoor3d_data(args.data_path, args.output_path)
    print(f'Processed data saved to {output_path}')
