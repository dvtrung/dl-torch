import multiprocessing
import os
import runpy
import time
from multiprocessing import Process
from typing import Dict, Tuple, Any

from dlex.datatypes import ModelReport
from dlex.train import get_unused_gpus
from dlex.utils import logger, table2str, logging

from .configs import Configs, Environment


manager = multiprocessing.Manager()
report_queue = manager.Queue()


def launch_evaluating(backend: str, params, configs, report=None):
    if backend is None:
        raise ValueError("No backend specified. Please add it in config file.")
    if backend == "sklearn":
        from dlex.sklearn.train import train
        train(params, configs, report)
        # runpy.run_module("dlex.sklearn.train", run_name=__name__)
    elif backend == "pytorch" or backend == "torch":
        from dlex.torch import PytorchBackend
        be = PytorchBackend(params, 0, report_queue)
        be.run_evaluate()
    elif backend == "tensorflow_v1":
        from dlex.tf.instance_v1 import TensorflowV1Backend
        be = TensorflowV1Backend(params)
        be.run_evaluate()
    elif backend == "tensorflow" or backend == "tf":
        runpy.run_module("dlex.tf.train", run_name=__name__)
    else:
        raise ValueError("Backend is not valid.")


def update_results(
        env: Environment,
        variable_values: Tuple[Any],
        report: ModelReport,
        all_reports: Dict[str, Dict[Tuple, ModelReport]],
        configs):
    """
    :param env:
    :param variable_values:
    :param report: an instance of ModelReport
    :param all_reports:
    :param configs:
    :return:
    """

    # all_reports are not always initialized
    if report:
        all_reports[env.name][variable_values] = report
    write_report(all_reports, configs)


def write_report(reports: Dict[str, Dict[Tuple, ModelReport]], configs):
    s = "\n# Report\n"

    #if report.param_details:
    #    s += f"\n## Model Summary\n\n{report.param_details}\n"
    #if report.num_params:
    #    s += f"\nNumber of parameters: {report.num_params:,}"
    #    s += f"\nNumber of trainable parameters: {report.num_trainable_params:,}\n"

    def _format_result(r: ModelReport, m: str):
        if r and r.results and m in r.results:
            return ("%.3f" % r.results[m]) + ("" if r.finished else " (running)")
        else:
            return ""

    for env in configs.environments:
        if env.name not in reports:
            continue
        s += f"\n## {env.title or env.name}\n"
        metrics = set.union(*[set(r.metrics or []) for r in reports[env.name].values() if r])
        metrics = list(metrics)
        if not env.report or env.report['type'] == 'raw':
            data = [env.variable_names + metrics]
            for variable_values, report in reports[env.name].items():
                data.append(
                    list(variable_values) +
                    [_format_result(report, metric) for metric in metrics])
            s += f"\n{table2str(data)}\n"
        elif env.report['type'] == 'table':
            val_row = env.variable_names.index(env.report['row'])
            val_col = env.variable_names.index(env.report['col'])
            for metric in metrics:
                s += "\nResults (metric: %s)\n" % metric
                data = [
                    [None for _ in range(len(env.variable_values[val_col]))]
                    for _ in range(len(env.variable_values[val_row]))
                ]
                for variable_values, report in reports[env.name].items():
                    _val_row = env.variable_values[val_row].index(variable_values[val_row])
                    _val_col = env.variable_values[val_col].index(variable_values[val_col])
                    if data[_val_row][_val_col] is None:
                        data[_val_row][_val_col] = _format_result(report, metric)
                    else:
                        data[_val_row][_val_col] += " / " + _format_result(report, metric)
                data = [[""] + env.variable_values[val_col]] + \
                    [[row_header] + row for row, row_header in zip(data, env.variable_values[val_row])]
                s += f"\n{table2str(data)}\n"

    print(s)
    if configs.args.gui:
        os.makedirs("model_reports", exist_ok=True)
        with open(os.path.join("model_reports", f"{configs.config_name}.md"), "w") as f:
            f.write(s)


def main():
    configs = Configs(mode="test")
    args = configs.args
    manager = multiprocessing.Manager()
    all_reports = manager.dict()

    envs = [e for e in configs.environments if e.name in configs.env_names]
    for env in envs:
        all_reports[env.name] = manager.dict()
        # init result list
        for variable_values, params in zip(env.variables_list, env.configs_list):
            all_reports[env.name][variable_values] = None

    gpu = args.gpu or get_unused_gpus(args)
    for env in envs:
        for variable_values, params in zip(env.variables_list, env.configs_list):
            params.gpu = gpu
            launch_evaluating(
                configs.backend, params, configs,
                report=None)


if __name__ == "__main__":
    main()