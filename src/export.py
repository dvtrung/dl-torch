import os
import shutil
from shutil import copyfile, rmtree, copytree

from configs import Configs

if __name__ == "__main__":
    configs = Configs(mode="export")
    params = configs.params
    working_dir = os.path.join("dist", params.path)
    rmtree(working_dir)
    os.makedirs(working_dir)

    src_files = [
        ("train.py", "train.py"),
        ("eval.py", "eval.py"),
        ("infer.py", "infer.py"),
        (os.path.join("models", params.model + ".py"), "model.py"),
        (os.path.join("datasets", params.dataset.name + ".py"), "dataset.py")
    ]
    for src, dst in src_files:
        copyfile(os.path.join("src", src), os.path.join(working_dir, dst))

    copytree(
        os.path.join("src", "utils"),
        os.path.join(working_dir, "utils"),
        ignore=shutil.ignore_patterns("__pycache__")
    )
