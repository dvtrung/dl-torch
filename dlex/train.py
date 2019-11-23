import multiprocessing
import os
import runpy
import time
from multiprocessing import Process
from threading import Thread
from typing import Dict, Tuple, Any

from dlex.utils import logger, table2str, logging
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
        params, args,
        variable_values: Tuple[Any],
        results, finished,
        all_results: Dict[str, Dict[Tuple, Any]],
        configs):
    if results is not None:
        all_results[env.name][variable_values] = {
            metric: ("%.3f" % res) + (" (running)" if not finished else "")
            for metric, res in results.items()}

    if args.report:
        os.system('clear')

    s = "# Report\n"
    for env in configs.environments:
        if env.name not in all_results:
            continue
        s += f"\n## {env.title or env.name}\n"
        if not env.report or env.report['type'] == 'raw':
            for vals, res in all_results.items():
                print("Configs: ", list(zip(env.variable_names, vals)))
                print("Result: ", res)
        elif env.report['type'] == 'table':
            results = all_results[env.name]
            val_row = env.variable_names.index(env.report['row'])
            val_col = env.variable_names.index(env.report['col'])
            for metric in params.test.metrics:
                s += "\nResults (metric: %s)\n" % metric
                data = [
                    [None for _ in range(len(env.variable_values[val_col]))]
                    for _ in range(len(env.variable_values[val_row]))
                ]
                for vals, ret in results.items():
                    _val_row = env.variable_values[val_row].index(vals[val_row])
                    _val_col = env.variable_values[val_col].index(vals[val_col])
                    if data[_val_row][_val_col] is None:
                        data[_val_row][_val_col] = ret[metric] if ret else ""
                    else:
                        data[_val_row][_val_col] += " / " + ret[metric]
                data = [[""] + env.variable_values[val_col]] + \
                    [[row_header] + row for row, row_header in zip(data, env.variable_values[val_row])]
                s += "\n" + table2str(data) + "\n"

    print(s)
    os.makedirs("model_reports", exist_ok=True)
    # with open(os.path.join("model_reports", f"{configs.config_name}.md"), "w") as f:
    #     f.write(s)


def main():
    configs = Configs(mode="train")
    args = configs.args
    manager = multiprocessing.Manager()
    all_results = manager.dict()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if configs.args.env:
        envs = [e for e in configs.environments if e.name in args.env]
    else:
        envs = [env for env in configs.environments if env.default]

    for env in envs:
        all_results[env.name] = manager.dict()
        # init result list
        for variable_values, params in zip(env.variables_list, env.parameters_list):
            all_results[env.name][variable_values] = None

    multi_processing = False
    if multi_processing:
        for env in envs:
            for variable_values, params in zip(env.variables_list, env.parameters_list):
                threads = []
                thread = Process(target=launch_training, args=(
                    configs.backend, params, args,
                    lambda ret, finished: update_results(
                        env, params, args,
                        variable_values,
                        ret, finished,
                        all_results,
                        configs)
                ))
                thread.start()
                time.sleep(5)
                threads.append(thread)
        for thread in threads:
            thread.join()
    else:
        for env in envs:
            for variable_values, params in zip(env.variables_list, env.parameters_list):
                launch_training(
                    configs.backend, params, args,
                    report_callback=lambda ret, finished: update_results(
                        env, params, args,
                        variable_values,
                        ret, finished,
                        all_results,
                        configs))


if __name__ == "__main__":
    main()