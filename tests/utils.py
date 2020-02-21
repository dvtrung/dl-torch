import tempfile

from dlex import Configs


def model_configs(yaml=None):
    def init_configs(func):
        def wrapper(*args, **kwargs):
            with tempfile.NamedTemporaryFile("w", suffix=".yml") as f:
                f.write(yaml)
                f.flush()
                configs = Configs("train", ["-c", f.name])
                func(configs)
        return wrapper
    return init_configs