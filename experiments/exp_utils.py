from pathlib import Path
import os

class Utils:
    @staticmethod
    def where_am_i() -> Path:
        """Return location of this file in full path(Jupyter Notebook can't use __file__)"""
        return Path(__file__).resolve().parent

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

