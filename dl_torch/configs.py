"""Reading model configurations"""

import os
import re
import argparse
import yaml


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
        return None

    def set(self, field, value):
        """Set key value."""
        setattr(self, field, value)

    def extend_default_keys(self, d):
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
        result_dir = os.path.join("model_outputs", self.path)
        return result_dir


def get_yaml_loader():
    """Get custom yaml loader that allows loading floating-point number"""
    yaml_loader = yaml.SafeLoader
    yaml_loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    return yaml_loader


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

    def __init__(self, mode, args=None):
        self.mode = mode
        self.parse_args(args)
        self.get_params()

    def parse_args(self, args=None):
        """Parse arguments."""
        parser = argparse.ArgumentParser(description="")

        parser.add_argument(
            '-c, --config_path',
            required=True,
            dest="config_path",
            help="path to model's configuration file")
        if self.mode == "train":
            parser.add_argument('--train', type=str2bool, nargs='?', const=True, default=True)
            parser.add_argument('--debug', action="store_true")
        parser.add_argument('--download', action="store_true", 
            help="Force to download, unzip and preprocess the data")
        parser.add_argument('--preprocess', action="store_true", 
            help="Force to preprocess the data")
        parser.add_argument('--verbose', action="store_true")
        parser.add_argument('-l, --load', dest="load", default=None, 
            required=self.mode in ["eval", "infer"],
            help="Tag of the checkpoint to load")
        if self.mode == "infer":
            parser.add_argument(
                '-i --input',
                nargs="*", action="append",
                dest="input")

        if args is None:
            self.args = parser.parse_args()
        else:
            self.args = parser.parse_args(args)

    def get_params(self):
        """Load model configs from yaml file"""
        with open(os.path.join("model_configs", self.args.config_path + ".yml"), 'r') as stream:
            try:
                params = yaml.load(stream, Loader=get_yaml_loader())
                params = AttrDict(params)
                params.set("mode", self.mode)
                params.set('path', self.args.config_path)
                self.params = params
            except yaml.YAMLError as exc:
                print(exc)
