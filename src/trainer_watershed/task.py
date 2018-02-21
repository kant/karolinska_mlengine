
import argparse
import os

import src.trainer_watershed.model as model
import src.trainer_watershed.input as input_pipe

import tensorflow as tf
from tensorflow.contrib.learn import Experiment, RunConfig
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import (saved_model_export_utils)
from tensorflow.contrib.training import HParams

def generate_experiment_fn(run_config, hparams, exp_param, input_param):
    """Create a method that constructs an experiment .

    Parameters
    ----------
    settings : dict
        A dictionary containing the desired experiment settings
    input_pipe_settings: dict
        A dictionary for settings of the input pipe.
        Note: When these are used for the eval input_fn, epochs are set to 1
            'batch_size' : int
                Batch size to return
            'epochs' : int
                How many epochs to return, None means infinite
            'num_parallel_calls' : int
                How many threads to use

    other_experiment_args:
        Adds possibility to affect other arguments of the experiment

    Returns
    -------
    Method
        Returns the method that returns the desired Experiment to a estimator
    """

    def _experiment_fn(run_config, hparams):

        return Experiment(
            model.build_estimator(run_config, hparams),
            train_input_fn=input_pipe.get_input_fn("train", **input_param),
            eval_input_fn=input_pipe.get_input_fn("val", **input_param),
            train_steps=exp_param['train_steps'],
            eval_steps=exp_param['eval_steps'],
            min_eval_frequency=exp_param['min_eval_frequency'])

    return _experiment_fn


if __name__ == '__main__':
    # To see available arguments:
    # https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='GCS or local path to training data',
        required=True
    )
    parser.add_argument(
        '--output_dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--model_type',
        help='Set which model type to use, see available in src.trainer_watershed.arch',
        required=True
    )
    parser.add_argument(
        '--export_only',
        help='Only export the model from the model_dir, ready for serving',
        default=False,
        type=bool
    )
    parser.add_argument(
        '--batch_size',
        help='Batch size to use',
        type=int,
        default=10
    )
    parser.add_argument(
        '--save_checkpoints_steps',
        help='How often to save a checkpoint',
        type=int,
        default=100)
    parser.add_argument(
        '--buffer_size',
        help='Buffer size to use',
        type=int,
        default=10
    )
    parser.add_argument(
        '--num_parallel_calls',
        help='How many samples in paralell to process in input pipe',
        default=None
    )
    parser.add_argument(
        '--train_epochs',
        help='Epochs to train for, default infinite',
        default=None
    )
    parser.add_argument(
        '--train_steps',
        help='Steps to run the training job for.',
        type=int,
        default=10000
    )
    parser.add_argument(
        '--eval_steps',
        help='Number of steps to run evalution for at each checkpoint',
        default=10,
        type=int
    )
    parser.add_argument(
        '--job-dir',
        help='this model ignores this field, but it is required by gcloud',
        default='junk'
    )
    parser.add_argument(
        '--eval_delay_secs',
        help='How long to wait before running first evaluation',
        default=None
    )
    parser.add_argument(
        '--min_eval_frequency',
        help='Minimum number of training steps between evaluations',
        default=100,
        type=int
    )
    parser.add_argument(
        '--save_summary_steps',
        help='Minimum steps of training steps between summaries',
        default=25,
        type=int
    )
    parser.add_argument(
        '--optimizer',
        help='Which optimizer to use, "Adam" or "RMS',
        default="Adam",
        type=str
    )
    parser.add_argument(
        '--learning_rate',
        help='Learning rate to use',
        default=0.0001,
        type=float
    )
    parser.add_argument(
        '--l2_gain',
        help='L2 regularization gain',
        default=0.0001,
        type=float
    )
    parser.add_argument(
        '--rmsprop_momentum',
        help='Rmsprop optimizer momentum',
        default=0.9,
        type=float
    )
    parser.add_argument(
        '--opt_epsilon',
        help='Epsilon term for the optimizer.',
        default=1.0,
        type=float
    )
    parser.add_argument(
        '--rmsprop_decay',
        help='Decay term for RMSProp.',
        default=0.9,
        type=float
    )
    parser.add_argument(
        '--unet_padding',
        help='What padding to use in u-net, SAME or VALID',
        default='SAME',
        type=str
    )

    args = parser.parse_args()

    # Input pipe settings
    input_param = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'buffer_size': args.buffer_size,
        'epochs': args.train_epochs,
        'num_parallel_calls': args.num_parallel_calls,
        'img_sizes': input_pipe.get_tf_record_image_size(args.data_dir),
        'padding': args.unet_padding
    }

    # Create run configuration default
    run_config = RunConfig()
    run_config = run_config.replace(model_dir=os.path.join(args.output_dir, args.model_type))
    run_config = run_config.replace(save_summary_steps=args.save_summary_steps)
    run_config = run_config.replace(save_checkpoints_steps=args.save_checkpoints_steps)

    # Define model and input parameters
    hparams = HParams(
        learning_rate=args.learning_rate,
        l2_gain=args.l2_gain,
        model_type=args.model_type,
        rmsprop_momentum=args.rmsprop_momentum,
        opt_epsilon=args.opt_epsilon,
        rmsprop_decay=args.rmsprop_decay,
        padding=args.unet_padding,
        optimizer=args.optimizer,
        model_dir=run_config.model_dir
    )

    # Define the experiment parameters
    exp_param = {
        'train_steps': args.train_steps,
        'eval_steps': args.eval_steps,
        'min_eval_frequency': args.min_eval_frequency
    }

    # Run the training job, only if this is not just an export job
    if not args.export_only:
        learn_runner.run(
            experiment_fn=generate_experiment_fn(run_config, hparams, exp_param, input_param),
            run_config=run_config,
            schedule="train_and_evaluate",
            hparams=hparams)

    # Export when done, but only if chief
    if run_config._is_chief:
        print("Master detected, exporting model..")
        export_predict_model = os.path.join(run_config.model_dir, 'predict_model')
        trained_estimator = model.build_estimator(run_config, hparams)
        trained_estimator.export_savedmodel(export_predict_model, model.serving_input_fn)
