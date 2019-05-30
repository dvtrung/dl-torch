"""Reading model configurations"""

import os
import re
import argparse
import yaml
import tempfile

from dlex.utils.logging import logger

DEFAULT_DATA_TMP_PATH = os.path.expanduser(os.path.join("~", "tmp"))


class ModuleConfigs:
    DATA_TMP_PATH = os.path.join(os.getenv("DATA_TMP_PATH", DEFAULT_DATA_TMP_PATH), "dlex", "datasets")


class AttrDict(dict):
    """Dictionary with key as property."""
    model = None
    dataset = None
    training_id = None
    seed = 1
    num_epochs = 30
    shuffle = False
    batch_size = None
    test_batch_size = None
    path = None

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key in self:
            if isinstance(self[key], dict):
                self[key] = AttrDict(self[key])

    def __getattr__(self, item):
        """
        :param str item:
        :rtype: AttrDict
        """
        logger.warning("Access to unset param %s", item)
        return None

    def set(self, field, value):
        """Set key value."""
        setattr(self, field, value)

    def extend_default_keys(self, d):
        """
        Add key and default values if not existed
        :param d: default key-value pairs
        :return:
        """
        for key in d:
            if isinstance(d[key], dict):
                if key in self:
                    self[key].extend_default_keys(d[key])
                else:
                    setattr(self, key, AttrDict(d[key]))
            else:
                if key not in self:
                    setattr(self, key, d[key])

    @property
    def log_dir(self):
        """Get logging directory based on model configs."""
        log_dir = os.path.join("logs", self.path)
        return os.path.join(log_dir, self.training_id)

    @property
    def output_dir(self):
        """Get output directory based on model configs"""
        result_dir = os.path.join("model_outputs", self.path)
        return result_dir


class Loader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(Loader, self).__init__(stream)

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            return yaml.load(f, Loader)


Loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.')
)


def str2bool(val):
    """Convert boolean argument."""
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Configs():
    """All configurations"""
    params = None
    args = None

    def __init__(self, mode, argv=None):
        self.mode = mode
        self.parse_args(argv)
        self.get_params()

    def parse_args(self, argv=None):
        """Parse arguments."""
        parser = argparse.ArgumentParser(description="")

        parser.add_argument(
            '-c, --config_path',
            required=True,
            dest="config_path",
            help="path to model's configuration file")
        if self.mode == "train":
            parser.add_argument('--debug', action="store_true",
                                help="train and eval on the same small data to check if the model works")
        parser.add_argument('--download', action="store_true",
                            help="force to download, unzip and preprocess the data")
        parser.add_argument('--preprocess', action="store_true",
                            help="force to preprocess the data")
        parser.add_argument('--verbose', action="store_true")
        parser.add_argument('-l, --load', dest="load", default=None,
                            required=self.mode in ["eval", "infer"],
                            help="tag of the checkpoint to load")
        parser.add_argument('--cpu', action='store_true', default=False,
                            help='disables CUDA training')
        if self.mode == "train":
            parser.add_argument('--num_processes', type=int, default=1, metavar='N',
                                help="how many training process to use")
            parser.add_argument('--save_all', action='store_true', default=False,
                                help='save every epoch')
        elif self.mode == "infer":
            parser.add_argument(
                '-i --input',
                nargs="*", action="append",
                dest="input")

        if argv is None:
            self.args = parser.parse_args()
        else:
            self.args = parser.parse_args(argv)

    def get_params(self):
        """Load model configs from yaml file"""
        path = os.path.join("model_configs", self.args.config_path + ".yml")
        try:
            with open(path, 'r') as stream:
                params = yaml.load(stream, Loader=Loader)
                params = AttrDict(params)
                params.set("mode", self.mode)
                params.set('path', self.args.config_path)
                self.params = params
        except yaml.YAMLError as exc:
            raise Exception("Invalid config syntax.")
        except FileNotFoundError:
            raise Exception("Config file '%s' not found." % path)
