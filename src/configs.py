import os
import yaml, re
import argparse
import sys

class AttrDict(dict):
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

    def add(self, field, value):
        setattr(self, field, value)

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

class Configs():
    params = None
    args = None

    def __init__(self):
        # self.parse_args()
        # self.get_params()
        pass

    def parse_args(self):
        parser = argparse.ArgumentParser(description="")

        parser.add_argument(
            '-c, --config_path',
            required=True,
            dest="config_path",
            help="path to model's configuration file")
        parser.add_argument('--debug', action="store_true")
        parser.add_argument('--verbose', action="store_true")
        parser.add_argument('--load', default=None)
        self.args = parser.parse_args()


    def get_params(self):
        with open(os.path.join("model_configs", self.args.config_path + ".yml"), 'r') as stream:
            try:
                params = yaml.load(stream, Loader=yaml_loader)
                self.params = AttrDict(params)
                self.params.add('path', self.args.config_path)
            except yaml.YAMLError as exc:
                print(exc)
