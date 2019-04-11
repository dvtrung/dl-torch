import os
import yaml, re
import argparse
import sys

class AttrDict(dict):
    model = None
    dataset = None
    training_id = None
    seed = 1
    num_epochs = 30
    shuffle = False
    batch_size = None
    test_batch_size = None

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key in self:
            if isinstance(self[key], dict):
                self[key] = AttrDict(self[key])

    def set(self, field, value):
        setattr(self, field, value)

    @property
    def log_dir(self):
        log_dir = os.path.join("logs", self.path)
        return os.path.join(log_dir, self.training_id)

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

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Configs():
    params = None
    args = None

    def __init__(self, mode, args=None):
        self.mode = mode
        self.parse_args(args)
        self.get_params()

    def parse_args(self, args=None):
        parser = argparse.ArgumentParser(description="")

        parser.add_argument(
            '-c, --config_path',
            required=True,
            dest="config_path",
            help="path to model's configuration file")
        if self.mode == "train":
            parser.add_argument('--train', type=str2bool, nargs='?', const=True, default=True)
            parser.add_argument('--debug', action="store_true")
        parser.add_argument('--force-preprocessing', action="store_true")
        parser.add_argument('--verbose', action="store_true")
        parser.add_argument('-l, --load', dest="load", default=None)
        if self.mode == "infer":
            parser.add_argument(
                '-i --input',
                nargs="*", action="append",
                dest="input",
                required=True)

        if args is None:
            self.args = parser.parse_args()
        else:
            self.args = parser.parse_args(args)

    def get_params(self):
        with open(os.path.join("model_configs", self.args.config_path + ".yml"), 'r') as stream:
            try:
                params = yaml.load(stream, Loader=yaml_loader)
                params = AttrDict(params)
                params.set("mode", self.mode)
                params.set('path', self.args.config_path)
                self.params = params
            except yaml.YAMLError as exc:
                print(exc)
