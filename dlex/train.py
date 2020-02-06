import logging
import multiprocessing
import os
import runpy
import shutil
import threading
import traceback
from collections import defaultdict
from datetime import datetime
from functools import partial
from time import sleep
from typing import Dict, Tuple, Any, List

from dlex.datatypes import ModelReport
from dlex.utils import logger, table2str, get_unused_gpus, subprocess, sys

from .configs import Configs, Environment

manager = multiprocessing.Manager()
all_reports: Dict[str, Dict[Tuple, ModelReport]] = manager.dict()
report_queue = manager.Queue()
short_report = None
long_report = None


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
        logger.error(e)
        logger.error(traceback.format_exc())
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


def _reduce_results(
        env: Environment,
        reduced_variable_names: List[str],
        metrics: List[str]) -> (List[str], Dict[Tuple, str]):
    variable_names = [name for name in env.variable_names if name not in reduced_variable_names]

    reports = defaultdict(lambda: [])
    for v_vals, report in all_reports[env.name].items():
        reduced_v_vals = tuple(
            [v_vals[i] for i, name in enumerate(env.variable_names) if name not in reduced_variable_names])
        reports[reduced_v_vals].append(report)

    results = defaultdict(lambda: [])
    infos = defaultdict(lambda: [])
    for reduced_v_vals in reports:
        results[reduced_v_vals] = [" ~ ".join([
            # f"{report.get_result_text(m, True)} [{report.training_idx}]" for report in reports[reduced_v_vals] if report
            f"{report.get_result_text(m, True)}" for report in reports[reduced_v_vals] if report
        ]) for m in metrics]
        infos[reduced_v_vals] = [' ~ '.join([report.get_status_text() for report in reports[reduced_v_vals] if report])]

    return variable_names, results, infos


def _short_report(s, s_long=None):
    global short_report, long_report
    short_report += f"{s}\n"
    long_report += f"{s_long or s}\n"


def _long_report(s):
    global long_report
    long_report += f"{s}\n"


def write_report():
    global short_report, long_report
    short_report = ""
    long_report = ""

    _short_report(f"# Report of {os.path.basename(configs.config_path)}")

    report_time = datetime.now()
    _short_report(f"\nLaunched at: %s" % launch_time.strftime('%b %d %Y %I:%M%p'))
    _short_report(f"Reported at: %s" % report_time.strftime('%b %d %Y %I:%M%p'))
    _short_report(f"Launch time: %s" % str(report_time - launch_time).split(".")[0])
    _short_report(f"Process id: %s" % os.getpid())

    _long_report(f"\nConfigs path: %s" % configs.config_path)
    _long_report(f"Log folder: %s" % configs.log_dir)

    _long_report(f"\n## Full configuration")
    _long_report(f"\n- Model: {str(configs.yaml_params.get('model'))}")
    _long_report(f"- Dataset: {str(configs.yaml_params.get('dataset'))}")
    _long_report(f"- Train: {str(configs.yaml_params.get('train'))}")
    _long_report(f"- Test: {str(configs.yaml_params.get('test'))}")

    for env in configs.environments:
        if env.name not in all_reports:
            continue
        _short_report(f"\n## {env.title or env.name}")
        metrics = _gather_metrics(all_reports[env.name])
        reduce = {name for name, vals in zip(env.variable_names, env.variable_values) if len(vals) <= 1}

        single_val = [f"\n- {name} = {vals[0]}" for name, vals in zip(env.variable_names, env.variable_values) if len(vals) == 1]
        if single_val:
            _short_report("\n### Configs: \n" + "".join(single_val))

        if not env.report or env.report['type'] == 'raw':
            variable_names, results, infos = _reduce_results(
                env,
                reduced_variable_names=list(set(env.report['reduce'] or []) | reduce),
                metrics=metrics)

            data = [variable_names + metrics + ["status"]]  # table headers
            for v_vals in results:
                data.append(list(v_vals) + results[v_vals] + infos[v_vals])
            _short_report(f"\n### Results\n\n{table2str(data)}")

            _long_report(f"\n### Details\n")
            for report in all_reports[env.name].values():
                if not report:
                    continue
                _long_report(
                    f"[{report.training_idx}] " + ", ".join([f"{m}: {report.get_result_text(m, full=True)}" for m in report.metrics]) + \
                    f"\n{report.param_details}"
                )

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
                        data[_val_row][_val_col] = report.get_result_text(metric)
                    else:
                        data[_val_row][_val_col] += " / " + report.get_result_text(metric)
                data = [[""] + env.variable_values[val_col]] + \
                       [[row_header] + row for row, row_header in zip(data, env.variable_values[val_row])]
                _short_report(f"\n{table2str(data)}\n")

    # logger.info(short_report)
    # if configs.args.report:
    #     refresh_display()

    with open(configs.report_path, "w") as f:
        f.write(long_report)

    return short_report, long_report


def get_training_idx(v_vals):
    return 0


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
    threading.Thread(target=refresh_display).start()
    # threading.Thread(target=listen_to_keyboard).start()

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
            report = report_queue.get()

            update_results(report, process_args[report.training_idx - 1][0].name, process_args[report.training_idx - 1][2])

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


def print_fixed_size(s, height, width):
    s = s.split('\n')
    row = sum([len(line) // width + 1 for line in s])
    s += [''] * (height - row)
    return '\n'.join(s)


def get_display_text():
    output = ""
    total_col, total_row = shutil.get_terminal_size()
    report_row = min(total_row - 5, int(9 * total_row / 10))
    log_row = total_row - report_row

    output += print_fixed_size(short_report, report_row, total_col)
    output += '-' * total_col

    f = subprocess.Popen(
        ['tail', '-n', str(log_row), os.path.join(configs.log_dir, "info.log")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output += print_fixed_size(f.stdout.read().decode().strip(), log_row, total_col)
    return output


def refresh_display():
    global configs, short_report
    if configs.args.report:
        while True:
            output = get_display_text()
            os.system('clear')
            print(output, end="")
            sleep(5)


def listen_to_keyboard():
    global configs, short_report
    if configs.args.report:
        while True:
            pass
            #key = sys.stdin.readline()
            #print(key)
            #output = get_display_text()
            #os.system('clear')
            #print(output, end="")
            #if key == ":":
            #    exit()


if __name__ == "__main__":
    launch_time = datetime.now()
    configs = Configs(mode="train")
    main()
