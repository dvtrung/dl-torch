import argparse
import glob
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        '-c, --config_path',
        required=True,
        dest="config_path",
        help="path to model's configuration file")

    parser.add_argument(
        '-l, --level',
        default='info',
        dest="level",
        help="log level")

    parser.add_argument(
        '--training_id',
        help="training_id")

    args = parser.parse_args()
    config_name = os.path.splitext(os.path.basename(args.config_path))[0]
    log_dir = os.path.join("logs", config_name)
    if args.training_id:
        log_dir = os.path.join(log_dir, *args.training_id.split('-'))
    else:
        log_dir = sorted(glob.glob(os.path.join(log_dir, "*-*")))[-1]
    os.system(f"clear; tail --retry -f {log_dir}/{args.level}.log")