import logging
import multiprocessing
import os
import runpy
import traceback
from collections import defaultdict
from datetime import datetime
from functools import partial
from time import sleep
from typing import Dict, Tuple, Any, List

import numpy as np
from dlex.datatypes import ModelReport
from dlex.utils import logger, table2str, get_unused_gpus

from .configs import Configs, Environment

manager = multiprocessing.Manager()
all_reports: Dict[str, Dict[Tuple, ModelReport]] = manager.dict()
report_queue = manager.Queue()


def launch_training(params, training_idx):
    backend = configs.backend

    if backend is None:
        raise ValueError("No backend specified. Please add it in config file.")

    try:
        if backend == "sklearn":
            from dlex.sklearn.train import train
            train(params, configs.args)
            # runpy.run_module("dlex.sklearn.train", run_name=__name__)
        elif backend == "pytorch" or backend == "torch":
            # runpy.run_module("dlex.torch.train", run_name=__name__)
            from dlex.torch.train import main
            return main(None, params, configs, training_idx, report_queue)
        elif backend == "tensorflow" or backend == "tf":
            runpy.run_module("dlex.tf.train", run_name=__name__)
        else:
            raise ValueError("Backend is not valid.")
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        if configs.args.notify:
            msg = "Error (%s): %s" % (configs.config_name, str(e))
            os.system(configs.args.notify_cmd % msg)


def update_results(
        report: ModelReport,
        env_name: str,
        variable_values: Tuple[Any]):
    """
    :param env_name:
    :param variable_values:
    :param report: an instance of ModelReport
    :return:
    """

    # all_reports are not always initialized
    if report:
        all_reports[env_name][variable_values] = report
    write_report()


def _error_callback(e: Exception):
    logger.error(str(e))
    logger.error(traceback.format_exc())


def _gather_metrics(report: Dict[Tuple, ModelReport]) -> List[str]:
    s = set()
    for r in report.values():
        if r and r.metrics:
            s |= set(r.metrics)
    return list(s)


def _format_result(r: ModelReport, m: str):
    if r and r.results and m in r.results:
        res = r.results[m]
        if r.finished:
            status = ""
        elif r.cv_num_folds:
            if type(res) == list and len(res) < r.cv_num_folds:
                status = f" (CV {r.cv_current_fold - 1}/{r.cv_num_folds})"
            else:
                status = ""
        else:
            status = f" (E {r.current_epoch - 1}/{r.num_epochs})"

        if res is None:
            result = "-"
        elif type(res) == float:
            result = "%.2f" % res
        elif type(res) == list and r.cv_num_folds:
            result = "[" + ", ".join(["%.2f" % r for r in res]) + "] -> " + "%.2f ± %.2f" % (np.mean(res), np.std(res))
        else:
            result = str(type(r.results[m]))
        return result + status
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
        reduced_v_vals = tuple(
            [v_vals[i] for i, name in enumerate(env.variable_names) if name not in reduced_variable_names])
        results[reduced_v_vals].append(report)

    for v_vals in results:
        results[v_vals] = [" ~ ".join([_format_result(report, m) for report in results[v_vals]]) for m in metrics]
    return variable_names, results


def write_report():
    if configs.args.report:
        # logger.setLevel(logging.NOTSET)
        os.system('clear')

    short_report = ""
    long_report = ""

    def _short_report(s):
        nonlocal short_report, long_report
        short_report += f"{s}\n"
        long_report += f"{s}\n"

    def _long_report(s):
        nonlocal long_report
        long_report += f"{s}\n"

    _short_report(f"# Report of {os.path.basename(configs.config_path)}")

    report_time = datetime.now()
    _short_report(f"\nLaunched at: %s" % launch_time.strftime('%b %d %Y %I:%M%p'))
    _short_report(f"Reported at: %s" % report_time.strftime('%b %d %Y %I:%M%p'))
    _short_report(f"Launch time: %s" % str(report_time - launch_time).split(".")[0])
    _short_report(f"Process id: %s" % os.getpid())

    _long_report(f"\nConfigs path: %s" % configs.config_path)
    _long_report(f"Log folder: %s" % configs.log_dir)

    for env in configs.environments:
        if env.name not in all_reports:
            continue
        _short_report(f"\n## {env.title or env.name}")
        metrics = _gather_metrics(all_reports[env.name])
        reduce = {name for name, vals in zip(env.variable_names, env.variable_values) if len(vals) <= 1}
        _short_report("\n### Configs: \n\n- " + \
                      "\n- ".join(
                          f"{name} = {vals[0]}" for name, vals in zip(env.variable_names, env.variable_values) if
                          len(vals) == 1))

        if not env.report or env.report['type'] == 'raw':
            variable_names, results = _reduce_results(
                env,
                reports=all_reports[env.name],
                reduced_variable_names=list(set(env.report['reduce'] or []) | reduce),
                metrics=metrics)

            data = [variable_names + metrics]  # table headers
            for v_vals in results:
                data.append(list(v_vals) + results[v_vals])
            _short_report(f"\n### Results\n\n{table2str(data)}")
        elif env.report['type'] == 'table':
            val_row = env.variable_names.index(env.report['row'])
            val_col = env.variable_names.index(env.report['col'])
            for metric in metrics:
                _short_report("\nResults (metric: %s)\n" % metric)
                data = [
                    [None for _ in range(len(env.variable_values[val_col]))]
                    for _ in range(len(env.variable_values[val_row]))
                ]
                for variable_values, report in all_reports[env.name].items():
                    _val_row = env.variable_values[val_row].index(variable_values[val_row])
                    _val_col = env.variable_values[val_col].index(variable_values[val_col])
                    if data[_val_row][_val_col] is None:
                        data[_val_row][_val_col] = _format_result(report, metric)
                    else:
                        data[_val_row][_val_col] += " / " + _format_result(report, metric)
                data = [[""] + env.variable_values[val_col]] + \
                       [[row_header] + row for row, row_header in zip(data, env.variable_values[val_row])]
                _short_report(f"\n{table2str(data)}\n")

    if configs.args.report:
        print(long_report)

    with open(os.path.join(configs.log_dir, "report.md"), "w") as f:
        f.write(short_report)
    with open(os.path.join(configs.log_dir, "report_full.md"), "w") as f:
        f.write(long_report)

    return short_report, long_report


def main():
    args = configs.args

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
            all_reports[env.name][variable_values] = None

    write_report()

    if args.num_processes >= 1:
        pool = multiprocessing.Pool(processes=args.num_processes)
        gpu = args.gpu or get_unused_gpus(args)
        results = []
        callbacks = []
        process_args = []
        for env in envs:
            for variable_values, params in zip(env.variables_list, env.configs_list):
                params.gpu = gpu
                callback = partial(update_results, env_name=env.name, variable_values=variable_values)
                callbacks.append(callback)
                process_args.append((env, params, variable_values))

        for idx, (pargs, callback) in enumerate(zip(process_args, callbacks)):
            _, params, _ = pargs
            r = pool.apply_async(
                launch_training,
                args=(params, idx + 1),
                callback=callback,
                error_callback=_error_callback)
            sleep(10)
            results.append(r)
        pool.close()

        while not all(r.ready() for r in results):
            idx, report = report_queue.get()
            update_results(report, process_args[idx - 1][0].name, process_args[idx - 1][2])

        # for r in results:
        #     r.get()
        pool.join()
    else:
        gpu = args.gpu or get_unused_gpus(args)
        idx = 0
        for env in envs:
            for variable_values, params in zip(env.variables_list, env.configs_list):
                idx += 1
                params.gpu = gpu
                report = launch_training(params, idx)
                update_results(report, env.name, variable_values)

    _, report = write_report()

    if args.notify:
        os.system(args.notify_cmd % report)

    # results.get(timeout=1)
    # for thread_args in threads_args:
    #     threads = []
    #     thread = Process(target=launch_training, args=thread_args)
    #     thread.start()
    #     time.sleep(5)
    #     threads.append(thread)

    # for thread in threads:
    #     thread.join()
    # else:
    #     gpu = args.gpu or get_unused_gpus(args)
    #     for thread_args in threads_args:
    #         params.gpu = gpu
    #         thread = Process(target=launch_training, args=thread_args)
    #         thread.start()
    #         thread.join()


if __name__ == "__main__":
    launch_time = datetime.now()
    configs = Configs(mode="train")
    main()
