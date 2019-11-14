import runpy
from typing import Dict, Tuple, Any, List

from dlex.utils import logger
from .configs import Configs


def launch_training(backend: str, params, args, report_callback=None):
    if backend is None:
        raise ValueError("No backend specified. Please add it in config file.")
    logger.info("Backend: %s", backend)
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
        variable_names: List[str],
        variable_values: Tuple[Any],
        results,
        all_results: Dict[Tuple, Any]):

    all_results[variable_values] = results
    metrics = list(results.keys())

    print("--------- REPORT ---------")
    for metric in metrics:
        print("(metric: %s)" % metric)
        if len(variable_names) == 2 and len(results) == 1:
            import pandas

            data = {}
            for vals, ret in all_results.items():
                if vals[0] not in data:
                    data[vals[0]] = {}
                if vals[1] not in data[vals[0]]:
                    data[vals[0]][vals[1]] = ret[metric] if ret else "-"
            print(pandas.DataFrame(data))
        else:
            for vals, res in all_results.items():
                print("Configs: ", list(zip(variable_names, vals)))
                print("Result: ", res)


def main():
    configs = Configs(mode="train")

    params_list, args = configs.params_list, configs.args

    all_results = {}
    variable_names = list(params_list[0][0].keys())

    for variable_values, params in params_list:
        all_results[tuple([variable_values[name] for name in variable_names])] = None

    for variable_values, params in params_list:
        launch_training(
            configs.backend, params, args,
            report_callback=lambda ret: update_results(
                variable_names,
                tuple([variable_values[name] for name in variable_names]),
                ret, all_results))


if __name__ == "__main__":
    main()