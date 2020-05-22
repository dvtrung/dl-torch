"""Reading model configurations"""
import argparse
import itertools
import os
import re
import tempfile
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Any, Union

import yaml
from dlex.utils.logging import set_log_dir, logger

DEFAULT_TMP_PATH = os.path.expanduser(os.path.join("~", "tmp"))
DEFAULT_DATASET_PATH = os.path.expanduser(os.path.join("~", "tmp", "datasets"))


class ModuleConfigs:
    @staticmethod
    def get_datasets_path():
        return os.getenv("DLEX_DATASET_PATH", DEFAULT_DATASET_PATH)

    @staticmethod
    def get_tmp_path():
        return os.path.join(os.getenv("DLEX_TMP_PATH", DEFAULT_TMP_PATH), "dlex")

    @staticmethod
    def get_checkpoint_dir():
        return os.getenv("DLEX_CHECKPOINT_PATH", "checkpoints")

    @staticmethod
    def get_log_dir():
        return os.getenv("DLEX_LOG_DIR", "logs")

    @staticmethod
    def get_output_dir():
        return os.getenv("DLEX_OUTPUT_DIR", "outputs")


class AttrDict(dict):
    _variables = None
    _overridden_params = None

    """Dictionary with key as property."""
    def __init__(self, *args, **kwargs):
        if '_variables' in kwargs:
            variables = kwargs['_variables']
            del kwargs['_variables']
        else:
            variables = {}

        overridden_params = None
        if '_overridden_params' in kwargs:
            overridden_params = kwargs['_overridden_params']  # type: dict
            del kwargs['_overridden_params']

        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

        if overridden_params:
            for key in overridden_params:
                if key not in self or not isinstance(self[key], dict):
                    self[key] = overridden_params[key]

        for key in self:
            if isinstance(self[key], Placeholder):
                if self[key].name in variables:
                    self[key] = variables[self[key].name]
                else:
                    self[key] = None
            elif isinstance(self[key], dict):
                self[key] = AttrDict(
                    self[key],
                    _variables=variables,
                    _overridden_params=overridden_params.get(key, None) if overridden_params else None)

        self._variables = variables
        self._overridden_params = overridden_params

    def __getattr__(self, item: str):
        logger.warning("Access to unset param %s", item)
        return None

    def set(self, prop: Union[str, list], value):
        """
        :param prop:
        :param value:
        """
        if isinstance(field, str):
            setattr(self, field, value)
        elif isinstance(field, list):
            if len(field) == 1:
                self.set(field, value)
            else:
                self[field[0]].set(field[1:], value)

    def recursively_set(self, prop: str, value: Any):
        """
        :param prop:
        :param value:
        """
        self.set(field.split('.'), value)

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
                    setattr(self, key, AttrDict(d[key], variables=self.variables))
            else:
                if key not in self:
                    setattr(self, key, d[key])

    def to_dict(self, level=1):
        d = {}
        for key in self:
            if key in ['_variables', '_overridden_params']:
                continue
            if isinstance(self[key], AttrDict) and level > 1:
                d[key] = self[key].to_dict(level=level - 1)
            elif isinstance(self[key], Placeholder) and self._variables:
                d[key] = self._variables[self[key].name]
            else:
                d[key] = self[key]
        return d

    def keys(self):
        return filter(lambda key: key not in ['_variables', '_overridden_params'], super().keys())

    def __iter__(self):
        return filter(lambda key: key not in ['_variables', '_overridden_params'], super().__iter__())

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


@dataclass
class OptimizerConfig:
    """
    Args:
        name (str): One of sgd, adam
    """
    name: str = "sgd"


@dataclass
class TrainConfig:
    """
    :param num_epochs: Number of epochs
    :type num_epochs: int
    :param batch_size: Batch size
    :type batch_size: int
    :param optimizer:
    :type optimizer: OptimizerConfig
    :param lr_scheduler (dict):
    :param eval: List of sets to be evaluated during training. Empty: no evaluation.
        Accepted values: `test`, `dev` (or `valid`).
        If both test and valid sets are presented, the test result for model with best valid result will also be recoreded. `dev` and `valid` can be used interchangeable
    :type eval: list
    :param max_grad_norm:
    :type max_grad_norm: float
    :param save_every: Time interval for saving model. Use s, m, h for number of seconds, minutes, hours. Use e for number of epochs.
            Examples: 100s, 30m, 2h, 1e
    :type save_every: str
    :param log_every: Time interval for logging to file
    :type log_every: str
    :param early_stop: Number of epochs to stop of results are not improving
    :type early_stop: int
    :param select_model: One of [best, last]
    """
    optimizer: OptimizerConfig
    num_epochs: int = None
    num_workers: int = None
    batch_size: int = None
    lr_scheduler: dict = None
    train_set: str = "train"
    valid_set: str = None
    max_grad_norm: float = 5.0
    save_every: str = "1e"
    log_every: str = None
    eval_every: str = "1e"
    cross_validation: int = None
    early_stop: int = None
    select_model: str = "best"
    show_report: bool = False
    show_progress: bool = False
    ema_decay_rate: float = None


@dataclass
class TestConfig:
    """
    :param batch_size:
    :type batch_size: int
    :param metrics: List of metrics for evaluation.
    :type metrics: list
    """
    batch_size: int = None
    metrics: List[str] = field(default_factory=lambda: ["acc"])
    log_every: str = "5s"
    output: str = None
    test_sets: List[str] = None


class Params(AttrDict):
    """Dictionary with key as property."""
    model = None
    tag = "default"
    random_seed = 1
    shuffle = False
    train: TrainConfig
    test: TestConfig
    gpu: List[int] = None
    training_idx = None

    def __init__(
            self,
            configs,
            env_name,
            yaml_configs,
            _variables: OrderedDict = None,
            _overridden_params: AttrDict = None):
        train_attr_dict = AttrDict(
            yaml_configs['train'],
            _variables=_variables,
            _overridden_params=_overridden_params.get('train', None))
        train = TrainConfig(**train_attr_dict.to_dict()) if "train" in yaml_configs else TrainConfig()

        test_attr_dict = AttrDict(
            yaml_configs['test'],
            _variables=_variables,
            _overridden_params=_overridden_params.get('test', None))
        test = TestConfig(**test_attr_dict.to_dict()) if "test" in yaml_configs else TestConfig()

        super().__init__(yaml_configs, _variables=_variables, _overridden_params=_overridden_params)

        self.train = train
        self.test = test
        self.configs = configs
        self.env_name = env_name
        self.variable_values = list(_variables.values()) if _variables else None

        # self.tag = ",".join(sorted([f"{name}={str(val)}" for name, val in _variables.items()])) if _variables else None
        self.tag = str(self.training_idx)

    @property
    def args(self):
        return self.configs.args

    @property
    def log_dir(self):
        return self.configs.log_dir

    @property
    def checkpoint_dir(self):
        path = self.configs.checkpoint_dir
        if self.env_name != "default":
            path = os.path.join(path, self.env_name)
            if self.tag:
                path = os.path.join(path, self.tag)
        return path


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
    list('-+0123456789.')
)


class Placeholder:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"[{self.name}]"


def add_variable_tag():
    tag = '!variable'
    Loader.add_implicit_resolver(tag, re.compile(u'''^~(.*)$'''), ['~'])

    def constructor_variables(loader, node):
        value = loader.construct_scalar(node)
        return Placeholder(value[1:])

    Loader.add_constructor(tag, constructor_variables)


add_variable_tag()


def str2bool(val):
    """Convert boolean argument."""
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


@dataclass
class Environment:
    name: str
    title: str = None
    default: bool = True

    # variables
    variable_names: List[str] = None
    variable_values: List[List[Any]] = None
    variables_list: List[Tuple[Any]] = None

    configs_list: List[Params] = None
    report: Any = None
    desc: str = None


class Configs:
    """All configurations"""
    args = None
    training_id = None

    def __init__(self, mode, argv=None):
        self.mode = mode
        self.parse_args(argv)

        # self.training_id = self.args.training_id or datetime.now().strftime('%Y%m%d-%H%M%S')
        self.training_id = datetime.now().strftime('%Y%m%d-%H%M%S')

        with open(self.config_path, 'r') as stream:
            try:
                self.yaml_params = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError:
                raise Exception("Invalid config syntax.")

        self.backend = self.yaml_params['backend']
        if self.args.configs:
            overridden_values = [s.split('=') for s in self.args.configs]

            def _override_value(d: dict, prop: list, val):
                if len(prop) == 1:
                    d[prop[0]] = val
                else:
                    _override_value(d[prop[0]], prop[1:], val)

            for key, value in overridden_values:
                key = key.split('.')
                _override_value(self.yaml_params, key, value)

        self._environments = []
        self.load_configs()

        # set_log_level(self.args.log)
        self.init_dirs()

    @classmethod
    def from_dict(cls):
        return cls()

    def init_dirs(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "results"), exist_ok=True)
        # shutil.rmtree(params.output_dir, ignore_errors=True)
        os.makedirs(self.output_dir, exist_ok=True)
        set_log_dir(self)

    @property
    def config_path(self):
        args = self.args
        paths = {
            args.config_path,
            os.path.join("model_configs", args.config_path),
            os.path.join("model_configs", args.config_path + ".yml")
        }
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            raise Exception("Config file '%s' not found." % args.config_path)
        else:
            assert len(paths) == 1
            return paths[0]

    @property
    def config_path_prefix(self):
        path = self.config_path
        if path[-4:] == ".yml":
            path = path[:-4]
        if path[:13] == "model_configs":
            path = path[14:]
        return path

    @property
    def config_name(self):
        return os.path.splitext(os.path.basename(self.config_path))[0]

    def parse_args(self, argv=None):
        """Parse arguments."""
        parser = argparse.ArgumentParser(description="")

        parser.add_argument(
            '-c, --config_path',
            required=True,
            dest="config_path",
            help="path to model's configuration file")
        parser.add_argument(
            '--configs',
            metavar="KEY=VALUE",
            nargs='+',
            required=False,
            dest="configs",
            help="Values to override configs read from file"
        )
        parser.add_argument(
            '--env', type=str, dest="env", help="List of environments")
        parser.add_argument(
            '--gui', action="store_true",
            help="GUI mode")
        parser.add_argument(
            '--show-logs',
            help="One of [none, debug, info, error, warn]", default="")
        parser.add_argument('-g, --gpus', nargs='+', help="Specify GPU(s) to use", dest='gpu')
        parser.add_argument('--num_gpus', help="Maximum number of GPUs to assign", type=int, default=1)
        parser.add_argument(
            '--gpu_memory_min',
            help="Minimum free memory (MiB) available for the device to be used", default=0)
        parser.add_argument(
            '--gpu_memory_max',
            help="Maximum used memory (MiB) available for the device to be used", default=100)
        # parser.add_argument('--training_id', help="Training ID")

        if self.mode == "train":
            parser.add_argument(
                '--debug', action="store_true",
                help="train and eval on the same small data to check if the model works")
            parser.add_argument(
                '--notify', action="store_true",
                help="send notification")
            parser.add_argument(
                '--notify-cmd', type=str, default="telegram-send --format markdown '```%s```'",
                help="command to run for sending notification")

        parser.add_argument('--download', action="store_true",
                            help="force to download, unzip and preprocess the data")
        parser.add_argument('--preprocess', action="store_true",
                            help="force to preprocess the data")
        parser.add_argument('--no-prepare', action="store_true",
                            help="do not prepare dataset")
        parser.add_argument('--verbose', action="store_true")
        parser.add_argument('-l, --load', dest="load", default=None,
                            required=self.mode in ["eval", "infer"],
                            help="tag of the checkpoint to load")
        parser.add_argument('--cpu', action='store_true', default=False,
                            help='disables CUDA training')

        parser.add_argument('--batch_size', default=None,
                            help="Size of each batch. This will overwrite the value in the config file")
        parser.add_argument(
            '--save_every', default=False,
            help='Save after a certain period of time. Unit: e (epoch), s, m, h (seconds, minutes, hours)')
        parser.add_argument(
            '--log_every', action='store_true', default=False,
            help='Log after a certain period of time. Unit: e (epoch), s, m, h (seconds, minutes, hours)')

        parser.add_argument('--num-workers', type=int, default=0, metavar='N',
                            help="Number of workers for loading data")

        parser.add_argument('--show-progress', action="store_true",
                            help="show progress bar")

        if self.mode == "train":
            parser.add_argument(
                '--checkpoint-tag', type=str, default=None,
                help="Tag of the checkpoint")
            parser.add_argument(
                '-p, --num-processes', type=int, default=0, metavar='N', dest='num_processes',
                help="number of training processes running at a time")
            parser.add_argument(
                '--save-all', action='store_true',
                help='save every epoch')
            parser.add_argument(
                '--exit-on-runtime-error', action="store_true",
                help="Exit when encoutering rumtime error (eg: CUDA out of memery). Exit code: 2")

            parser.add_argument(
                '--output_test_samples', action="store_true",
                help="Output samples after evaluation."
            )

            parser.add_argument(
                "--test-set", default=None, dest="test_sets", nargs='+',
                help="Set to evaluate"
            )
        elif self.mode == "test":
            parser.add_argument(
                "--set", default=None, dest="test_sets", nargs='+',
                help="Set to evaluate"
            )
        elif self.mode == "infer":
            parser.add_argument(
                '-i --input',
                nargs="*", action="append",
                dest="input")
        elif self.mode == "log":
            parser.add_argument(
                '--level',
                type=str, default="info",
                help="log level")

        if argv is None:
            self.args, _ = parser.parse_known_args()
        else:
            self.args, _ = parser.parse_known_args(argv)

    @property
    def environments(self) -> List[Environment]:
        return self._environments

    @property
    def env_names(self) -> List[str]:
        if self.args.env:
            return sorted(self.args.env.split(' '))
        else:
            return ['default']

    @property
    def log_dir(self):
        """Get logging directory based on model configs."""
        env_name = "_".join(self.env_names)
        if env_name == "default":
            log_dir = os.path.join(ModuleConfigs.get_log_dir(), self.config_name)
        else:
            log_dir = os.path.join(ModuleConfigs.get_log_dir(), self.config_name, env_name)

        if self.training_id:
            return os.path.join(log_dir, *self.training_id.split('-'))
        else:
            return log_dir

    @property
    def checkpoint_dir(self):
        return os.path.join(ModuleConfigs.get_checkpoint_dir(), self.config_name)

    @property
    def output_dir(self):
        """Get output directory based on model configs"""
        return os.path.join(ModuleConfigs.get_output_dir(), self.config_name)

    @property
    def report_path(self):
        if not os.path.exists("model_reports"):
            os.makedirs("model_reports")
        return os.path.join(
            "model_reports",
            f"{self.config_name}_{'_'.join(self.env_names)}_{self.training_id.replace('-', '_')}.md")

    def create_params(self, env_name, variables=None, overridden_params: dict = None) -> Params:
        params = Params(
            self,
            env_name,
            self.yaml_params,
            _variables=variables,
            _overridden_params=overridden_params or {})

        args = self.args
        params.test.test_sets = args.test_sets or params.test.test_sets

        return params

    def load_configs(self):
        if 'env' in self.yaml_params:
            for env_name in self.env_names:
                if env_name not in self.yaml_params['env'].keys():
                    logger.warn("Environment %s not found" % env_name)

            for env_name, env_prop in self.yaml_params['env'].items():
                if env_name not in self.env_names:
                    continue

                # Unfold params for every variable combination
                configs_list = []  # there are multiple configs if variables are chosen from lists

                variables_list = []
                variable_names = list(env_prop['variables'].keys()) if 'variables' in env_prop else []
                variable_values = list(env_prop['variables'].values()) if 'variables' in env_prop else []
                overridden_params = env_prop['params'] if 'params' in env_prop else dict()

                # fill default values
                if 'default' in self.yaml_params['env'] and 'variables' in self.yaml_params['env']['default']:
                    vals = self.yaml_params['env']['default']['variables']
                    for name, val in vals.items():
                        if name not in variable_names:
                            variable_names.append(name)
                            variable_values.append(val)

                variable_values = [val if isinstance(val, list) else [val] for val in variable_values]

                for variable_values_combination in itertools.product(*variable_values):
                    variables = OrderedDict()
                    for name, val in zip(variable_names, variable_values_combination):
                        variables[name] = val
                    params = self.create_params(env_name, variables, overridden_params)

                    variables_list.append(tuple([variables[name] for name in variable_names]))
                    configs_list.append(params)

                report = {}

                # Parse report properties
                if 'report' in env_prop:
                    report['type'] = env_prop['report']['type']
                    report['reduce'] = env_prop['report']['reduce']
                    if report['type'] == 'table':
                        report['row'] = env_prop['report']['row']
                        report['col'] = env_prop['report']['col']
                else:
                    report['reduce'] = []
                    if False and len(variable_names) == 2:
                        report['type'] = 'table'
                        report['row'] = variable_names[0]
                        report['col'] = variable_names[1]
                    else:
                        report['type'] = 'raw'

                self._environments.append(Environment(
                    name=env_name,
                    default=env_prop.get('default', True),
                    title=env_prop.get('title', env_name),
                    desc=env_prop.get('desc', None),
                    variable_names=variable_names,
                    variable_values=variable_values,
                    variables_list=variables_list,
                    configs_list=configs_list,
                    report=report
                ))
        else:
            params = self.create_params("default")
            params.mode = self.mode
            params.config_path = self.config_path
            params.tag = "default"

            self._environments.append(Environment(
                name="default",
                title="Default",
                variable_names=[],
                variable_values=[],
                variables_list=[tuple()],
                configs_list=[params],
                report=dict(type='raw', reduce=[])
            ))

    def get_default_params(self):
        return self.environments[0].configs_list[0]


def yaml_configs(content: str, argv=None):
    """
    Decorator to load configs from string
    :param content:
    :param argv:
    :return:
    """
    def init_configs(func):
        def wrapper(*args, **kwargs):
            with tempfile.NamedTemporaryFile("w", suffix=".yml") as f:
                f.write(content)
                f.flush()
                configs = Configs("train", ["-c", f.name, *(argv or [])])
                func(*args, **kwargs, configs=configs)
        return wrapper
    return init_configs


class YamlConfigs:
    def __init__(self, content, argv):
        f = tempfile.NamedTemporaryFile("w", suffix=".yml")
        f.write(content)
        f.flush()
        self.configs = Configs("train", ["-c", f.name, *(argv or [])])
        self.f = f

    def __enter__(self):
        return self.configs

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()