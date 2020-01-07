import multiprocessing
import os
import runpy
import time
from collections import defaultdict
from multiprocessing import Process
from typing import Dict, Tuple, Any, Set, List

from dlex.datatypes import ModelReport
from dlex.utils import logger, table2str
import logging

from .configs import Configs, Environment


def launch_training(backend: str, params, args, report_callback=None):
    if backend is None:
        raise ValueError("No backend specified. Please add it in config file.")
    if backend == "sklearn":
        from dlex.sklearn.train import train
        train(params, args, report_callback)
        # runpy.run_module("dlex.sklearn.train", run_name=__name__)
    elif backend == "pytorch" or backend == "torch":
        # runpy.run_module("dlex.torch.train", run_name=__name__)
        from dlex.torch.train import main
        main(None, params, args, report_callback)
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
    :param hyperparam_values:
    :param report: an instance of ModelReport
    :param all_reports:
    :param configs:
    :return:
    """

    # all_reports are not always initialized
    if report:
        all_reports[env.name][variable_values] = report
    write_report(all_reports, configs)


def _gather_metrics(report: Dict[Tuple, ModelReport]) -> List[str]:
    s = set()
    for r in report.values():
        if r and r.metrics:
            s |= set(r.metrics)
    return list(s)


def _format_result(r: ModelReport, m: str):
    if r and r.results and m in r.results:
        if r.finished:
            status = ""
        elif r.cross_validation_num_folds:
            status = f" (cross validation {r.cross_validation_current_fold}/{r.cross_validation_num_folds})"
        else:
            status = f" (epoch {r.current_epoch}/{r.num_epochs})"
        return f"%.3f{status}" % r.results[m]
    else:
        return ""


def _reduce_results(
        env: Environment,
        reports: Dict[Tuple, ModelReport],
        reduced_variable_names: List[str],
        metrics: List[str]):
    variable_names = [name for name in env.variable_names if name not in reduced_variable_names]
    results = defaultdict(lambda: [])
    for v_vals, report in reports.items():
        reduced_v_vals = tuple([v_vals[i] for i, name in enumerate(env.variable_names) if name not in reduced_variable_names])
        results[reduced_v_vals].append(report)

    for v_vals in results:
        results[v_vals] = [" ~ ".join([_format_result(report, m) for report in results[v_vals]]) for m in metrics]
    return variable_names, results


def write_report(reports: Dict[str, Dict[Tuple, ModelReport]], configs):
    if configs.args.report:
        logger.setLevel(logging.NOTSET)
        os.system('clear')

    s = "\n# Report\n"

    #if report.param_details:
    #    s += f"\n## Model Summary\n\n{report.param_details}\n"
    #if report.num_params:
    #    s += f"\nNumber of parameters: {report.num_params:,}"
    #    s += f"\nNumber of trainable parameters: {report.num_trainable_params:,}\n"

    for env in configs.environments:
        if env.name not in reports:
            continue
        s += f"\n## {env.title or env.name}\n"
        metrics = _gather_metrics(reports[env.name])
        if not env.report or env.report['type'] == 'raw':
            variable_names, results = _reduce_results(
                env,
                reports=reports[env.name],
                reduced_variable_names=env.report['reduce'] or [],
                metrics=metrics)

            data = [variable_names + metrics]  # table headers
            for v_vals in results:
                data.append(list(v_vals) + results[v_vals])
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

    if configs.args.report:
        print(s)
        os.makedirs("model_reports", exist_ok=True)
        with open(os.path.join("model_reports", f"{configs.config_name}.md"), "w") as f:
            f.write(s)


def get_unused_gpus(args):
    try:
        logger.info("Searching for unused GPUs...")
        import subprocess
        from io import StringIO
        import pandas as pd

        gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        gpu_df = pd.read_csv(StringIO(gpu_stats.decode()), names=['memory.used', 'memory.free'], skiprows=1)
        gpu_df['memory.used'] = gpu_df['memory.used'].map(lambda x: int(x.rstrip(' [MiB]')))
        gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
        gpu_df = gpu_df[gpu_df['memory.used'] <= args.gpu_memory_max]
        gpu_df = gpu_df[gpu_df['memory.free'] >= args.gpu_memory_min]
        idx = gpu_df.index.tolist()[:args.num_gpus]
        return idx
    except Exception:
        return []


def main():
    configs = Configs(mode="train")
    args = configs.args
    manager = multiprocessing.Manager()
    all_reports = manager.dict()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if configs.args.env:
        envs = [e for e in configs.environments if e.name in args.env]
    else:
        envs = [env for env in configs.environments if env.default]

    for env in envs:
        all_reports[env.name] = manager.dict()
        # init result list
        for variable_values, params in zip(env.variables_list, env.configs_list):
            all_reports[env.name][variable_values] = manager.dict()

    multi_processing = False
    if multi_processing:
        for env in envs:
            for variable_values, params in \
                    zip(env.variables_list, env.configs_list):
                threads = []
                thread = Process(target=launch_training, args=(
                    configs.backend, params, args,
                    lambda report: update_results(
                        env, variable_values, report, all_reports, configs)
                ))
                thread.start()
                time.sleep(5)
                threads.append(thread)
        for thread in threads:
            thread.join()
    else:
        gpu = args.gpu or get_unused_gpus(args)
        for env in envs:
            for variable_values, params in zip(env.variables_list, env.configs_list):
                params.gpu = gpu
                launch_training(
                    configs.backend, params, args,
                    report_callback=lambda report: update_results(
                        env, variable_values, report, all_reports, configs))


if __name__ == "__main__":
    main()