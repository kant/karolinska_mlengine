import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
import src.lib.tfops as tfops

def simple(features, mode, hparams, scope='simple_network'):
    """Returns a simple network architecture.

    conv[5,5,32] -> Dense -> Dropout -> deconv[5,5,2]

    Parameters
    ----------
    features : Tensor
        4D Tensor where the first dimension is the batch size, then height, width
        and channels
    mode : tensorflow.python.estimator.model_fn.ModeKeys
        Class that contains the current mode
    scope : str, optional
        The scope to use for this architecture

    Returns
    -------
    Tensor op
        Return the final tensor operation (logits), from the network
    """
    with tf.variable_scope(scope):
        is_training = mode == Modes.TRAIN
        raise Exception('Implement this')

        return net[-1]
