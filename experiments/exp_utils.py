from pathlib import Path
import os
import sys

import pandas as pd

class PointnetPath:
    """Hold paths for pointnet parameters.
    Minimum usage:
        this_path = GenPath().HERE

    Code map:
    - qwa-work/                  : QW_REPO
      - data/                    : DATA_ROOT
        - pointnet-log/          : log_root
          - <experiment name>/   : log_dir (fold name string)
      - pointnet-pytorch-jupyter : PN_REPO
        - data/
          - <experiment name>/   : data_dir
        - experiments/           : HERE
          - exp_utils.py         : Path(__file__)
    """

    def __init__(self, exp_name: str):
        self.HERE = PointnetPath.where_am_i()
        self.log_dir = exp_name

        # pointnet-pytorch-jupyter repo
        self.PN_REPO = Path(self.HERE, "..").resolve()
        assert self.PN_REPO.exists(), "Pointnet repo doesn't exist."

        # qwa-work repo
        self.QW_REPO = Path(self.PN_REPO, "..").resolve()
        assert self.QW_REPO.exists(), "qwa-work repo doesn't exist."

        # qwa-work/data
        self.DATA_ROOT = self.QW_REPO / "data"
        assert self.DATA_ROOT.exists(), "qwa-work/data doesn't exist."

        # qwa-work/data/pointnet-log
        self.log_root = self.DATA_ROOT / "pointnet-log"
        self.log_root.mkdir(exist_ok=True)

        # pointnet-pytorch-jupyter/data
        self.data_dir = self.PN_REPO / "data" / self.log_dir
        self.data_dir.parent.mkdir(exist_ok=True)

    @staticmethod
    def where_am_i() -> Path:
        """Return location of this file in full path(Jupyter Notebook can't use __file__)"""
        return Path(__file__).resolve().parent

class PointCloudTable:

    @staticmethod
    def prep(
        pc_df: pd.DataFrame, cols: list[str], label_map: dict[int, int], drop_label: int = 24
    ) -> pd.DataFrame | None:
        """Pass pandas dataframe and return only specified columns.

        Parameters
        ----------
        pc_df: pandas.DataFrame
            Point cloud data with x,y,z,nx,ny,nz,r,g,b,h,s,v,l,a,b,label,uuid
        cols: list[str]
            List of columns to return. Anything else is dropped.
        label_map: dict[int, int]
            Mapping of label number. {input label: output label}
        drop_label: int
            If pc_df["label"] has drop_label, return. Default 24 is "leaves-fruit".
        """
        if drop_label is not None:
            if drop_label in pc_df["label"].unique():
                return
        return pc_df.replace(label_map)[cols]

class JumpDir:
    """
    change directory for reading/saving files, importing modules and other operations and return to the original directory

    Parameters
    ----------
    target_path : str | pathlib.Path
        path to change to
        can be relative path

    start_path : str | pathlib.Path
        path of starting returning directory
        absulute path is recommended

    Example
    -------
    for jupyter notebook, use os.getcwd() instead of __file__
    jupyter notebook cannnot know the location of the script so manually check your path

    with JumpDir(os.path.join('..', 'paht', 'to', 'lib'), os.path.dirname(os.path.abspath(__file__))):
        contents = os.listdir()

    """

    def __init__(self, target_path: str | Path, start_path: str | Path):
        self.dest = Path(target_path)
        self.home = Path(start_path)

    def __enter__(self):
        # check if current directory is the location of the script
        if Path(os.getcwd()) != self.home:
            os.chdir(self.home)
            print(f"Current directory is not starting path; cd {self.home}")

        try:
            os.chdir(self.dest)
        except FileNotFoundError as err:
            print(f"path does not exist: {self.dest}")
            print(f"your current path  : {os.getcwd()}")
            raise err

    def __exit__(self, exc_type, exc_value, traceback):
        os.chdir(self.home)

class AddPath:
    """
    safely add a path to sys.path for importing modules

    sys.path.insert might not add the path if it is already there
    and sys.path.remove might remove the wrong path if it occurs

    reference:
    https://stackoverflow.com/questions/17211078/how-to-temporarily-modify-sys-path-in-python

    Parameters
    ----------
    path : str | pathlib.Path
        path to add to sys.path
        always absolute path

    Example
    -------
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'path', 'to', 'lib')
    with AddPath(lib_path):
        module = __import__('mymodule')

    """

    def __init__(self, path: str | Path):
        self.path = str(Path(path).resolve())

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass

