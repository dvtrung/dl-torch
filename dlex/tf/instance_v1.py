import logging
import os
import random
from collections import OrderedDict, namedtuple
from datetime import datetime

import tensorflow.compat.v1 as tf
from dlex import FrameworkBackend, TrainingProgress
from dlex.configs import Params
from dlex.datasets.tf import Dataset
from dlex.tf.models.base_v1 import BaseModelV1
from dlex.tf.utils.model_utils import get_model
from dlex.utils import get_num_iters_from_interval, get_num_seconds_from_interval, Datasets
from dlex.utils.logging import logger
from dlex.utils.model_utils import get_dataset
from tensorflow.estimator import LoggingTensorHook, CheckpointSaverListener, \
    EstimatorSpec, TrainSpec, EvalSpec
from tqdm import tqdm

tf.disable_v2_behavior()


EvaluationResults = namedtuple("EvaluationResults", "results outputs")


class TensorflowV1Backend(FrameworkBackend):
    def __init__(
            self,
            params: Params = None,
            training_idx: int = None,
            report_queue=None):
        super().__init__(params, training_idx, report_queue)
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logger.info(f"Training started ({training_idx}).")

        # tf.enable_eager_execution()
        # tf.random.set_random_seed(params.seed)

        # X_train, y_train = dataset_train.load_data()
        # X_test, y_test = dataset_test.load_data()
        # train_generator = ImageDataGenerator(
        #    rescale=1.0/255, horizontal_flip=True,
        #    width_shift_range=4.0/32.0, height_shift_range=4.0/32.0)
        # test_generator = ImageDataGenerator(rescale=1.0/255)
        # y_train = to_categorical(y_train)
        # y_test = to_categorical(y_test)

    def load_model(self, mode) -> (BaseModelV1, Datasets):
        """
        Load model and dataset
        :param mode: train, test, dev
        :param report:
        :param argv:
        :param params: if None, configs will be read from file
        :param args:
        :return:
        """

        args = self.configs.args
        params = self.params

        # Init dataset
        dataset_builder = get_dataset(params)
        assert dataset_builder, "Dataset not found."
        if not args.no_prepare:
            dataset_builder.prepare(download=args.download, preprocess=args.preprocess)

        datasets = Datasets(
            "tensorflow", dataset_builder,
            train_set=params.train.train_set,
            valid_set=params.train.valid_set,
            test_sets=params.test.test_sets)

        # Init model
        model_cls = get_model(params)
        model = model_cls(params, datasets.train_set)  # type: BaseModelV1

        # model.summary()

        # log model summary
        # parameter_details = [["Name", "Shape", "Trainable"]]
        # num_params = 0
        # num_trainable_params = 0
        # for n in tf.get_default_graph().as_graph_def().node:
        #     parameter_details.append([
        #         n.name,
        #         "test",
        #         "✓" if False else ""])
        #     num_params += np.prod(list(parameter.shape))
        #     if parameter.requires_grad:
        #         num_trainable_params += np.prod(list(parameter.shape))

        # s = table2str(parameter_details)
        # logger.debug(f"Model parameters\n{s}")
        # logger.debug(" - ".join([
        #     f"No. parameters: {num_params:,}",
        #     f"No. trainable parameters: {num_trainable_params:,}"
        # ]))
        # report.param_details = s
        # report.num_params = num_params
        # report.num_trainable_params = num_trainable_params

        # use_cuda = torch.cuda.is_available()
        # if use_cuda and params.gpu:
        #     gpus = [f"cuda:{g}" for g in params.gpu]
        #     model = DataParellelModel(model, gpus)
        #     logger.info("Start training using %d GPU(s): %s", len(params.gpu), str(params.gpu))
        #     torch.cuda.set_device(torch.device(gpus[0]))
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #     model.to(gpus[0])
        # else:
        #     model = DataParellelModel(model, ['cpu'])

        # logger.debug("Dataset: %s. Model: %s", str(dataset_builder), str(model_cls))
        # if use_cuda:
        #     logger.info("CUDA available: %s", torch.cuda.get_device_name(0))

        return model, datasets

    def run_train(self) -> None:
        self.train_with_session()

    def run_evaluate(self) -> None:
        params = self.params
        model, datasets = self.load_model("test")
        model.build_graph()

        saver = tf.train.Saver(max_to_keep=1)
        if params.train.ema_decay_rate:
            ema_saver = tf.train.Saver(model.ema.variables_to_restore(), max_to_keep=5)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            local_init = tf.local_variables_initializer()
            sess.run(local_init)

            model.load_checkpoint(sess, saver, tag="latest")
            model.load_checkpoint(sess, ema_saver, tag="latest")
            for name, dataset in datasets.test_sets.items():
                res = self.evaluate_with_session(
                    sess,
                    model,
                    dataset,
                    output_path=os.path.join(params.log_dir, "results"),
                    output_tag=f"evaluate_val")
                logger.info(f"[{name}]: {str(res.results)}")

    def train_with_session(self) -> None:
        params = self.params
        args = self.args
        model, datasets = self.load_model("train")

        model.build_graph()
        logger.info("Successfully built model")
        logger.info("Training metrics: %s", list(model.metric_ops.keys()))

        step = 0
        if self.args.debug:
            next_debug_step = 0
            debug_ops = model.get_debug_ops()

        saver = tf.train.Saver(
            max_to_keep=1)
        if params.train.ema_decay_rate:
            ema_saver = tf.train.Saver(model.ema.variables_to_restore(), max_to_keep=5)

        self.report.set_model_summary(
            variable_names=[var.name for var in tf.trainable_variables()],
            variable_shapes=[var.shape.as_list() for var in tf.trainable_variables()],
            variable_trainable=[var.trainable for var in tf.trainable_variables()]
        )

        prog = TrainingProgress(params, len(datasets.train_set))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            local_init = tf.local_variables_initializer()
            global_init = tf.global_variables_initializer()
            sess.graph.finalize()
            sess.run(global_init)

            if args.load:
                model.load_checkpoint(args.load, sess, saver)
                logger.info("Loaded checkpoint: %s", args.load)

            for epoch in range(1, params.train.num_epochs + 1):
                sess.run(local_init)
                prog.new_epoch(epoch)

                model.set_training(True)
                batch_size = self.params.train.batch_size * len(params.gpu)

                if self.params.dataset.shuffle:
                    datasets.train_set.shuffle()
                data = datasets.train_set.data
                batches = []
                for batch_start in range(0, len(data), batch_size):
                    batches.append(data[batch_start:batch_start + batch_size])
                # if self.params.shuffle:
                #     random.shuffle(batches)
                with tqdm(total=len(data), desc=f"Epoch {epoch}") as t:
                    for batch in batches:
                        feed = {}
                        datasets.train_set.populate_feed_dict(feed, model.placeholders, batch)
                        model.populate_feed_dict(feed, is_training=True)

                        _, global_step, *metrics = sess.run([
                            model.train_op,
                            model.global_step,
                            *list(model.metric_ops.values())], feed_dict=feed)

                        if self.args.debug and step >= next_debug_step:
                            vals = sess.run(list(debug_ops.values()), feed_dict=feed)
                            for name, val in zip(debug_ops.keys(), vals):
                                logger.debug(f"{name}\n{val}")
                                input()
                            num_steps = input("Number of steps do you want to run (default: 1): ") or 1
                            next_debug_step += int(num_steps)

                        t.update(len(batch))
                        prog.update(len(batch))
                        t.set_postfix(dict(
                            **{key: val[0] for key, val in zip(model.metric_ops.keys(), metrics)}
                        ))
                        step += 1

                        # Save model
                        if prog.should_save():
                            model.save_checkpoint(sess, saver, "latest")

                        # Log
                        if prog.should_log():
                            logger.info(", ".join([
                                f"epoch: {epoch}",
                                f"global step: {global_step}",
                                f"progress: {int(prog.epoch_progress * 100)}%",
                                *[f"{name}: {val[0]:.4f}" for name, val in zip(model.metric_ops.keys(), metrics)]
                            ]))

                model.save_checkpoint(sess, saver, "latest")
                # model.load_checkpoint(sess, saver, "latest")
                for name, dataset in datasets.test_sets.items():
                    res = self.evaluate_with_session(
                        sess,
                        model,
                        dataset,
                        output_path=params.checkpoint_dir,
                        output_tag="latest")
                    self.report.add_epoch_results(res.results)
                    self.update_report()
                    logger.info(res.results)
        self.report.results = self.report.current_results
        self.update_report()
        return self.report

    def evaluate_with_session(
            self,
            sess,
            model: BaseModelV1,
            dataset: Dataset,
            output_path: str = None,
            output_tag: str = None) -> EvaluationResults:
        batch_size = self.params.train.batch_size
        data = dataset.data

        all_preds = []
        all_refs = []

        batches = []
        outputs = []
        for batch_start in range(0, len(data), batch_size):
            batches.append(data[batch_start:batch_start + batch_size])

        with tqdm(total=len(data), desc=f"Eval") as t:
            for batch in batches:
                feed = {}
                dataset.populate_feed_dict(feed, model.placeholders, batch)
                model.populate_feed_dict(feed, is_training=False)
                pred, ref, *metrics = sess.run(
                    [model.predictions, model.references, *list(model.metric_ops.values())],
                    feed_dict=feed)
                pred = pred if type(pred) == list else list(pred)
                ref = ref if type(ref) == list else list(ref)
                assert len(pred) == len(ref) == len(batch)
                all_preds += pred
                all_refs += ref
                t.update(len(batch))
                t.set_postfix(**{key: val[0] for key, val in zip(model.metric_ops.keys(), metrics)})

                for p, b in zip(pred, batch):
                    str_input, str_ground_truth, str_predicted = dataset.format_output(p, b)
                    outputs.append(dict(
                        input=str_input,
                        reference=str_ground_truth,
                        hypothesis=str_predicted))
                    # logger.debug(outputs[-1])

        results = {}
        for metric in self.params.test.metrics:
            results[metric] = dataset.evaluate(all_preds, all_refs, metric, output_path)

        if self.params.test.output and output_path:
            path = dataset.write_results_to_file(
                all_preds,
                # sample_ids,
                output_path,
                output_tag,
                self.params.test.output)
            dataset.builder.run_evaluation_script(path)

        for output in random.choices(outputs, k=20):
            logger.debug(output)

        return EvaluationResults(
            results={key: results[key] for key in results},
            outputs=outputs)

    def train_with_estimator(self):
        run_config = tf.estimator.RunConfig(
            model_dir=self.params.checkpoint_dir,
            save_checkpoints_steps=get_num_iters_from_interval(self.params.train.save_every),
            save_checkpoints_secs=get_num_seconds_from_interval(self.params.train.save_every),
            save_summary_steps=100,
            keep_checkpoint_max=1
        )

        def model_fn(features, labels, mode, params):
            output = self.model.forward(features)
            loss = self.model.get_loss(features, output)
            train_op = self.model.get_train_op(loss)
            metric_ops = self.model.get_metric_ops(features, output)
            return EstimatorSpec(
                mode=mode, loss=loss,
                train_op=train_op,
                eval_metric_ops=metric_ops,
                training_hooks=[
                    TqdmHook(OrderedDict(loss=loss), len(self.datasets.train_set), params['batch_size']),
                    tf.estimator.LoggingTensorHook(dict(loss=loss), every_n_iter=10)
                ],
                evaluation_hooks=[
                    TqdmHook(OrderedDict(metrics=metric_ops['acc']), len(self.datasets.test), params['batch_size'])
                ])

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            params=dict(batch_size=self.params.train.batch_size),
            config=run_config)
        self.report.launch_time = datetime.now()
        num_train_steps = int(len(self.datasets.train_set) / self.params.train.batch_size * self.params.train.num_epochs)

        train_spec = TrainSpec(
            input_fn=self.datasets.train_set.input_fn,
            max_steps=num_train_steps)
        eval_spec = EvalSpec(
            input_fn=self.datasets.test.input_fn,
            steps=5,
            start_delay_secs=150,
            throttle_secs=200
        )
        logger.debug(train_spec)

        logger.info("Training started.")
        # estimator.train(
        #     input_fn=datasets.train._input_fn,
        #     max_steps=num_train_steps,
        #     hooks=[
        #         TqdmHook(model.loss, len(datasets.train), params.train.batch_size)
        #     ]
        # )
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        logger.info("Training done.")

    def set_seed(self):
        super().set_seed()
        tf.set_random_seed(self.params.random_seed)


class TqdmHook(tf.estimator.SessionRunHook):
    def __init__(self, postfix: OrderedDict, total, batch_size):
        self.postfix = postfix
        # self._timer = SecondOrStepTimer(every_steps=1)
        self._should_trigger = False
        self._iter_count = 0
        self._pbar = None
        self.total = total
        self.batch_size = batch_size

    def begin(self):
        pass
        # self._timer.reset()

    @property
    def pbar(self):
        if not self._pbar:
            self._pbar = tqdm(desc="Train", total=self.total)
        return self._pbar

    def before_run(self, run_context):
        # self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
        return tf.estimator.SessionRunArgs(dict(
            global_step=tf.train.get_or_create_global_step(), **self.postfix))

    def after_run(self, run_context, run_values):
        # if self._should_trigger:
        res = run_values.results
        self.pbar.update(self.batch_size)
        if self.pbar.n > self.total:
            self.pbar.n = self.pbar.n % self.total
        self.pbar.set_description("Epoch %d" % ((res['global_step'] * self.batch_size) // self.total + 1))
        pf = OrderedDict({name: str(res[name]) for name in self.postfix})
        self.pbar.set_postfix(pf)
        self.pbar.refresh()


class EvalLogHook(LoggingTensorHook):
    def __init__(self, metric_ops):
        super().__init__()
        self.metric_ops = metric_ops

    def begin(self):
        super().begin()

    def after_run(self, run_context, run_values):
        super().after_run(run_context, run_values)
        logger.debug(run_values)


class CheckpointSaverListenerEx(CheckpointSaverListener):
    def __init__(self):
        pass

    def begin(self):
        pass

    def after_save(self, session, global_step_value):
        logger.info("Checkpoint saved.")
        # logger.info("Evaluating model...")
        # results = self.estimator.evaluate(
        #     input_fn=self.datasets.test.input_fn,
        #     steps=None)
        # logger.debug(str(results))