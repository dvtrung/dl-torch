import curses
import logging
import multiprocessing
import os
import shutil
import sys
import threading
import traceback
from collections import defaultdict
from datetime import datetime
from functools import partial
from multiprocessing import Queue
from time import sleep
from typing import Dict, Tuple, List, Union

from dlex.datatypes import ModelReport
from dlex.utils import logger, table2str, get_unused_gpus
from dlex.utils.curses import CursesManager
from dlex.utils.tmux import TmuxManager

from .configs import Configs, Environment

LOG_WINDOWS_HEIGHT = 10

manager = multiprocessing.Manager()
all_reports: Dict[int, Union[ModelReport, None]] = {}
report_queue = manager.Queue()
short_report = None
long_report = None


def launch_training(params, training_idx):
    backend = configs.backend

    if backend is None:
        raise ValueError("No backend specified. Please add it in config file.")

    if backend == "sklearn":
        from dlex.sklearn.train import train
        train(params, configs.args)
        # runpy.run_module("dlex.sklearn.train", run_name=__name__)
    elif backend == "pytorch" or backend == "torch":
        from dlex.torch import PytorchBackend
        be = PytorchBackend(params, training_idx, report_queue)
        return be.run_train()
    elif backend == "tensorflow_v1" or backend == "tf_v1":
        from dlex.tf.instance_v1 import TensorflowV1Backend
        be = TensorflowV1Backend(params, training_idx, report_queue)
        return be.run_train()
    elif backend == "tensorflow" or backend == "tf":
        from dlex.tf.instance_v2 import TensorflowV2Backend
        be = TensorflowV2Backend(params, training_idx, report_queue)
        return be.run_train()
    elif backend == "tff":
        from dlex.tf.tff import TensorflowFederatedBackend
        be = TensorflowFederatedBackend(params, training_idx, report_queue)
        return be.run_train()
    else:
        raise ValueError("Backend is not valid.")

    #except Exception as e:
    #    logger.error(e)
    #    logger.error(traceback.format_exc())
    #    if configs.args.notify:
    #        msg = "Error (%s): %s" % (configs.config_name, str(e))
    #        os.system(configs.args.notify_cmd % msg)


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
    for training_idx, report in all_reports.items():
        if report is None:
            continue
        v_vals = report.params.variable_values
        reduced_v_vals = tuple(
            [v_vals[i] for i, name in enumerate(env.variable_names) if name not in reduced_variable_names])
        reports[reduced_v_vals].append(report)

    results = defaultdict(lambda: [])
    infos = defaultdict(lambda: [])
    for reduced_v_vals in reports:
        results[reduced_v_vals] = [" ~ ".join([
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
        for training_idx in all_reports:
            all_reports[training_idx] = ModelReport.load(configs.log_dir, training_idx)

        _short_report(f"\n## {env.title or env.name}")
        metrics = _gather_metrics(all_reports)
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
            for report in all_reports.values():
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
                for report in all_reports.values():
                    variable_values = report.params.variable_values
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
    if configs.args.gui:
        refresh_display()

    with open(configs.report_path, "w") as f:
        f.write(long_report)

    return short_report, long_report


def get_training_idx(v_vals):
    return 0


def _exit():
    pool.terminate()
    pool.join()
    if configs.args.gui:
        tmux.close_all_panes()
    sys.exit()


def _add_thread(name, target, args=()):
    threads[name] = threading.Thread(target=target, args=args)
    threads[name].setDaemon(True)
    threads[name].start()


def _on_key_pressed(c):
    if c in ["i", "d"]:
        log = dict(i="info", d="debug")[c]
        if tmux.is_tmux_session:
            if tmux.is_pane_visible(log):
                tmux.close_pane(log)
                for name in ['info', 'debug']:
                    if tmux.is_pane_visible(name):
                        tmux.resize_pane(name, height=LOG_WINDOWS_HEIGHT)
            else:
                tmux.split_window(
                    log, cmd=f"tail --retry -f {configs.log_dir}/{log}.log",
                    height=LOG_WINDOWS_HEIGHT)
        else:
            # def refresh_fn():
            #     return "test"
            # curses.show_info_dialog(refresh_fn)
            curses.show_info_dialog(["tail", "--retry", "-n", "100", "-f", f"{configs.log_dir}/{log}.log"])
    elif c == "g":
        def refresh_fn():
            return "test"
        curses.show_info_dialog(refresh_fn)
    elif c == "m":
        selection = curses.select(dict(
            i="[i] Show/Hide Info Log",
            d="[d] Show/Hide Debug Log",
            g="[g] GPU Usage",
            t="[t] Training Details",
            q="[q] Exit Training"
        ), "i", title="Menu")
        _on_key_pressed(selection)
    elif c == "r":
        write_report()
    elif c == "q":
        _exit()


def main(scr=None, *args):
    args = configs.args
    # tmux.split_window(f"tail --retry -f {configs.log_dir}/info.log")
    if scr:
        # _on_key_pressed("i")
        # _on_key_pressed("d")
        pass
    if args.debug:
        logger.setLevel(logging.DEBUG)

    def _refresh_display():
        while True:
            write_report()
            sleep(5)
    _add_thread("refresh_display", _refresh_display)

    if args.num_processes >= 1:
        gpu = args.gpu or get_unused_gpus(args)
        results = []
        callbacks = []
        process_args = []

        for env in configs.environments:
            for idx, (variable_values, params) in enumerate(zip(env.variables_list, env.configs_list)):
                params.gpu = [gpu[idx % len(gpu)]]
                callback = partial(update_results, env_name=env.name, training_idx=idx)
                callbacks.append(callback)
                process_args.append((env.name, params, idx))
                all_reports[idx] = None

        for pargs, callback in zip(process_args, callbacks):
            env_name, params, idx = pargs
            process_names.put(f"{env_name}-{idx}")
            r = pool.apply_async(
                launch_training,
                args=(params, idx),
                callback=callback,
                error_callback=_error_callback)
            # sleep(10)
            results.append(r)

        # pool.close()

        def _update_results():
            while not all(res.ready() for res in results):
                report = report_queue.get()
                env_name, _, training_idx = process_args[report.training_idx]
                update_results(report, env_name, training_idx)
                refresh_display()
        _add_thread("update_results", _update_results)

        while True:
            try:
                c = scr.getkey()
                if c:
                    _on_key_pressed(c)
            except (KeyboardInterrupt, SystemExit):
                _exit()
            except:
                pass
            # if c == "m":
            #     build_menu(tmux, scr)

        # for r in results:
        #     r.get()
        # pool.join()
    else:
        gpu = args.gpu or get_unused_gpus(args)
        training_idx = 0
        for env in configs.environments:
            for variable_values, params in zip(env.variables_list, env.configs_list):
                training_idx += 1
                params.gpu = gpu
                launch_training(params, training_idx)
                write_report()

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
    output += print_fixed_size(short_report, total_row, total_col)

    return output


def refresh_display():
    if configs.args.gui:
        curses.main_text = get_display_text()
        curses.refresh(clear=True)


def update_results(report: ModelReport, env_name: str, training_idx: int):
    logger.debug("Results updated (env: %s, process id: %d)", env_name, training_idx)
    logger.debug(report.current_test_results)
    # all_reports[env_name][training_idx] = report
    write_report()


def signal_handler(signal, frame):
    print("Program killed")
    # _exit()


if __name__ == "__main__":
    # signal.signal(signal.SIGINT, signal_handler)
    launch_time = datetime.now()
    tmux = TmuxManager()
    curses = CursesManager()
    configs = Configs(mode="train")
    process_names = Queue()
    if configs.args.num_processes:
        def initialize(names):
            multiprocessing.current_process().name = str(names.get())
        pool = multiprocessing.Pool(
            processes=configs.args.num_processes,
            initializer=initialize,
            initargs=(process_names,))
    threads = {}
    if configs.args.gui:
        curses.wrapper(main)
    else:
        main()