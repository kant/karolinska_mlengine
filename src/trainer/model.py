import os
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
from importlib import import_module
from src.lib.tfops import batch_base64_to_tensor

def _cnn_model_fn(features, labels, mode, params):
    """Creates the model function.

    This will handle all the different processes needed when using an Estimator.
    The estimator will change "modes" using the mode flag, and depending on that
    different outputs are provided.

    Parameters
    ----------
    features : Tensor
        4D Tensor where the first dimension is the batch size, then height, width
        and channels
    labels : Dict {'label': Tensor, 'weight': Tensor}
        Contains both weight and label, where each is a 3D Tensor, where the first dimension is
        the batch size, then height and width. The values in the label image is class number, while
        weight is a weight map for the pixels
    mode : tensorflow.python.estimator.model_fn.ModeKeys
        Class that contains the current mode
    params : class
        Contains all the hyper parameters that are available to the model. These can be different
        depending on which architecture (model type) is in use

    Returns
    -------
    tf.estimator.EstimatorSpec
        The requested estimator spec
    """

    feature_input = features['inputs']
    # Fetch the desired architecture
    module = import_module('src.trainer.arch')
    model_arch = getattr(module, params.model_type)

    # Logits Layer
    logits = model_arch(feature_input, mode, params)

    # If this is a prediction or evaluation mode, then we return
    # the class probabilities and the guessed pixel class
    if mode in (Modes.TRAIN, Modes.EVAL, Modes.PREDICT):
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')
        predicted_pixels = tf.argmax(input=logits, axis=-1)

    # During training and evaluation, we calculate the loss
    if mode in (Modes.TRAIN, Modes.EVAL):
        global_step = tf.train.get_or_create_global_step()
        label_indices = tf.cast(labels['label'], tf.int32)
        softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels['label'],
                                                                 logits=logits)
        weighted_softmax = tf.multiply(softmax, labels['weight'])
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = tf.reduce_sum(weighted_softmax) + reg_loss

        tf.summary.scalar('OptimizeLoss', loss)
        tf.summary.scalar('RegLoss', reg_loss)
        tf.summary.image('Feature', feature_input)
        tf.summary.image('Label', tf.expand_dims(tf.cast(labels['label'] * 255, tf.uint8), 3))
        tf.summary.image('Weight', tf.expand_dims(tf.cast(labels['weight'] * 255, tf.uint8), 3))
        pred_estend = tf.cast(tf.expand_dims(predicted_pixels * 255, 3), tf.uint8)
        tf.summary.image('Prediction', pred_estend)

    # When predicting (running inference only, during serving for example) we
    # need to return the output as a dictionary.
    if mode == Modes.PREDICT:
        predictions = {
            'classes': predicted_pixels,
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)

    # In training (not evaluation) we perform backprop
    if mode == Modes.TRAIN:
        if params.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        elif params.optimizer == 'RMS':
            optimizer = tf.train.RMSPropOptimizer(
                params.learning_rate,
                decay=params.rmsprop_decay,
                momentum=params.rmsprop_momentum,
                epsilon=params.opt_epsilon)
        else:
            raise Exception('Unknown optimizer specified.')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op)

    # If evaluating only, we perform evaluation operations
    if mode == Modes.EVAL:
        # Accuracy operations
        eval_metric_ops = {
            'accuracy': tf.metrics.mean_iou(label_indices, predicted_pixels, 2)
        }

        # Create a SummarySaverHook
        eval_summary_hook = tf.train.SummarySaverHook(
            save_steps=1,
            output_dir=os.path.join(params.model_dir, 'eval'),
            summary_op=tf.summary.merge_all())

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
            evaluation_hooks=[eval_summary_hook])


def build_estimator(run_config, hparams):
    """Build the estimator using the desired model type

    Parameters
    ----------
    model_dir : str
        The directory to store the saved model in
    model_type : str
        The type of model to use, see src.trainer.arch
    learning_rate : float
        The desired learning rate

    Returns
    -------
    Estimator object
        A higher level API that simplifies training and evaluation of neural networks.
    """
    return tf.estimator.Estimator(model_fn=_cnn_model_fn,
                                  model_dir=run_config.model_dir,
                                  config=run_config,
                                  params=hparams)


def parse_incoming_tensors(incoming):
    img = tf.reshape(incoming, [-1, 512, 512, 1])
    return tf.cast(img, tf.float32) * (1. / 255) - 0.5


def serving_input_fn():
    """Input function to use when serving the model."""
    inputs = tf.placeholder(tf.string, shape=(None, ))
    feature_input = batch_base64_to_tensor(inputs)
    feature_input = {'inputs': parse_incoming_tensors(feature_input)}

    return tf.estimator.export.ServingInputReceiver(feature_input, inputs)
