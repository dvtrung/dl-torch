import os

from dlex.configs import Configs

if __name__ == "__main__":
    configs = Configs("log")

    args = configs.args
    config_name = os.path.splitext(os.path.basename(args.config_path))[0]
    log_dir = configs.log_dir
    os.system(f"clear; tail --retry -f {log_dir}/{args.level}.log")