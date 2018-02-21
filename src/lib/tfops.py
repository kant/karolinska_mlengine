import tensorflow as tf
import numpy as np

def deconv2d_resize(inputs,
                    filters,
                    kernel_size=(2, 2),
                    padding='SAME',
                    strides=(2, 2),
                    reuse=None,
                    name=None,
                    activation=None):
    """Resize input using nearest neighbor then apply convolution.

    Parameters
    ----------
    inputs : tensor
        The input tensor to this operation
    filters : int
        Number of filters of the conv operation
    kernel_size : tuple, optional
        The kernel size to use
    padding : str, optional
        Padding strategy
    strides : tuple, optional
        How many steps the resize operation should take, the strides
        control how big the output tensor is
    reuse : None, optional
        Variable to control if the generated weights should be reused from somewhere else
    name : None, optional
        Desired name of this op
    activation : None, optional
        Desired activation function

    Returns
    -------
    tensor
        The output tensor that has been resized and convolved
    """
    shape = inputs.get_shape().as_list()
    height = shape[1] * strides[0]
    width = shape[2] * strides[1]
    resized = tf.image.resize_nearest_neighbor(inputs, [height, width])

    return tf.layers.conv2d(resized, filters,
                            kernel_size=kernel_size,
                            padding='SAME',
                            strides=(1, 1),
                            reuse=reuse,
                            name=name,
                            activation=activation)

def conv_l2_bn_relu(inputs,
                    filters,
                    kernel_size,
                    padding='VALID',
                    strides=(1, 1),
                    reuse=None,
                    name=None,
                    activation=tf.nn.relu,
                    is_training=False,
                    l2_gain=0.0001):
    """A convolution with L2 regularisation, batch norm and relu as output activation.

    Parameters
    ----------
    inputs : tensor
        The input tensor to this operation
    filters : int
        Number of filters of the conv operation
    kernel_size : tuple, optional
        The kernel size to use
    padding : str, optional
        Padding strategy
    strides : tuple, optional
        How many steps the resize operation should take, the strides
        control how big the output tensor is
    reuse : None, optional
        Variable to control if the generated weights should be reused from somewhere else
    name : None, optional
        Desired name of this op
    activation : tf.nn.relu, optional
        Desired activation function
    is_training : bool, optional
        Used to turn on/off the batch normalization update
    l2_gain: float, optional
        The L2 regularizer gain to use

    Returns
    -------
    tensor
        The output tensor with the conv, l2, bn, relu operations added
    """
    net = [inputs]
    sigma = np.sqrt(2.0 / (filters * kernel_size[0] * kernel_size[1]))

    net.append(tf.layers.conv2d(
        net[-1],
        filters, kernel_size,
        activation=None,
        padding=padding,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_gain),
        name="conv_%s" % name))

    net.append(tf.layers.batch_normalization(net[-1], training=is_training, name="bn_%s" % name))
    net.append(activation(net[-1], name="relu_%s" % name))

    return net[-1]


def base64_to_tensor_image(websafe_base64):
    return tf.image.decode_image(tf.decode_base64(websafe_base64))


def batch_base64_to_tensor(input_text):
    img = tf.map_fn(tf.image.decode_image, tf.decode_base64(input_text), dtype=tf.uint8)
    return img


def crop_and_concat(x1, x2):
    with tf.variable_scope('cropped_concat'):
        x1_shape = x1.get_shape().as_list()
        x2_shape = x2.get_shape().as_list()
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)


def pad_to_desired_3D(tensor, shape):
    tensor_shape = tensor.get_shape().as_list()[0:2]
    add_dim = [(float(a) - b) / 2 for a, b in zip(shape, tensor_shape)]
    for i, dim in enumerate(add_dim):
        msg = "dim %d: Cannot centralize the image, check that the size adds up" % i
        assert (dim).is_integer(), msg
    pad_list = [[int(dim), int(dim)] for dim in add_dim]
    paddings = tf.constant(pad_list)

    def apply_padding(tensor):
        return tf.pad(tensor, paddings, "REFLECT")

    return tf.transpose(tf.map_fn(apply_padding, tf.transpose(tensor)))
