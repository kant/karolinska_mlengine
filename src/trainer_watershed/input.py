from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
from src.lib.fileops import get_all_files_containing
from src.lib.tfops import pad_to_desired_3D

tf.logging.set_verbosity(tf.logging.INFO)

def parse_single_example(record):
    """Parses a single feature."""
    feature = {'feature/img': tf.FixedLenFeature([], tf.string),
               'feature/height': tf.FixedLenFeature([], tf.int64),
               'feature/width': tf.FixedLenFeature([], tf.int64),
               'feature/channels': tf.FixedLenFeature([], tf.int64),
               'label/img': tf.FixedLenFeature([], tf.string),
               'label/height': tf.FixedLenFeature([], tf.int64),
               'label/width': tf.FixedLenFeature([], tf.int64),
               'label/channels': tf.FixedLenFeature([], tf.int64),
               'weight/img': tf.FixedLenFeature([], tf.string),
               'weight/height': tf.FixedLenFeature([], tf.int64),
               'weight/width': tf.FixedLenFeature([], tf.int64),
               'weight/channels': tf.FixedLenFeature([], tf.int64),
               'energy/img': tf.FixedLenFeature([], tf.string)
               }

    return tf.parse_single_example(record, feature)


def read_and_decode(record, ftr_shape, lbl_shape, padding):
    """Given a a tf.TFRecordReader().read output, return the decoded sample.

    The sizes of the tensor need to be "hardcoded" when creating a TF graph.
    This is "bypassed" by running a seperate session in the main file that
    extracts the size of the image from the tf record directly. This is
    generally not done though.

    Parameters
    ----------
    record : serialized_example
        A TensorFlow serialized_example read from a tfrecord

    Returns
    -------
    feature_img: Tensor
        The feature image to train with
    label_img
        The label image
    """
    parsed = parse_single_example(record)

    # Get the sizes
    ftr_height = tf.cast(parsed['feature/height'], tf.int32)
    ftr_width = tf.cast(parsed['feature/width'], tf.int32)
    ftr_channel = tf.cast(parsed['feature/channels'], tf.int32)
    lbl_height = tf.cast(parsed['label/height'], tf.int32)
    lbl_width = tf.cast(parsed['label/width'], tf.int32)
    lbl_channel = tf.cast(parsed['label/channels'], tf.int32)
    wgt_height = tf.cast(parsed['label/height'], tf.int32)
    wgt_width = tf.cast(parsed['label/width'], tf.int32)
    wgt_channel = tf.cast(parsed['label/channels'], tf.int32)

    # read, decode and normalize image
    feature_img = tf.decode_raw(parsed['feature/img'], tf.uint8)
    feature_img = tf.cast(feature_img, tf.float32) * (1. / 255) - 0.5
    feature_img = tf.reshape(feature_img, ftr_shape)
    label_img = tf.decode_raw(parsed['label/img'], tf.uint8)
    label_img = tf.cast(label_img, tf.float32)
    label_img = tf.reshape(label_img, lbl_shape)
    weight_img = tf.decode_raw(parsed['weight/img'], tf.uint8)
    weight_img = tf.cast(weight_img, tf.float32) / 255.0
    weight_img = tf.reshape(weight_img, lbl_shape)
    weight_img = tf.squeeze(weight_img)
    energy_img = tf.decode_raw(parsed['energy/img'], tf.uint8)
    energy_img = tf.cast(energy_img, tf.int32, name='helo')
    energy_img = tf.reshape(energy_img, lbl_shape)
    energy_img = tf.squeeze(energy_img)

    # We need to increase the size of the input image, if padding is set to "VALID"
    if padding == 'VALID':
        feature_img = tf.reshape(
            pad_to_desired_3D(feature_img, [780, 780, 1]),
            [780, 780, 1]
        )

    return feature_img, label_img, energy_img

def get_input_feeder(tfrecords, params):
    """Creates a input feeder method.

    The estimator uses the returned method to create a input node on the graph.

    Parameters
    ----------
    tfrecords : list
        List of tfrecords
    params : dict
        batch_size : int
            Batch size to return
        buffer_size: int,
            How much to buffer up
        epochs : int
            How many epochs to return, None means infinite
        num_parallel_calls : int
            How many threads to use

    Returns
    -------
    tensor
        A input feeder generator
    """

    # Wrap the decode method
    def wrapped_read_and_decode(record):
        return read_and_decode(
            record,
            params['img_sizes'][0],
            params['img_sizes'][1],
            params['padding'])

    def get_next():
        """Create the input feeder next method.

        Note, the feature output is wrapped within a dictionary to support future serving
        of the model
        """
        # Create a filename queue for the TFRecords
        dataset = tf.data.TFRecordDataset(tfrecords)

        # Parse the record into tensors.
        dataset = dataset.map(wrapped_read_and_decode,
                              num_parallel_calls=params['num_parallel_calls'])
        dataset = dataset.prefetch(params["buffer_size"])
        dataset = dataset.shuffle(params["buffer_size"])
        # If no epochs are provided, we repeat the dataset indefinetly
        if params['epochs'] is None:
            dataset = dataset.repeat()  # Repeat the input indefinitely.
        else:
            dataset = dataset.repeat(params['epochs'])
        dataset = dataset.batch(params["batch_size"])
        iterator = dataset.make_one_shot_iterator()
        next_example, next_mask, next_energy = iterator.get_next()

        return {'inputs': next_example, 'mask': next_mask}, {'label': next_energy}

    return get_next

def get_input_fn(input_type, data_dir, **kwargs):
    """Creates the correct input function based on the desired input type.

    Parameters
    ----------
    input_type : str
        A string, one of 'train', 'eval', 'test'
    data_dir : str
        Full path to the directory containing the tfrecord files
    batch_size=2: int, optional
        Batch size to use
    buffer_size=2: int, optional
        Buffer size
    epochs=None: int, optional
        How many epochs should the input fn yield
    num_parallel_calls=None: int, optional
        How many threads to spawn for the input pipe

    Returns
    -------
    Generator
        Returns an input feeder generator of the desired type
    """
    assert input_type in ['train', 'val', 'test']
    params = {
        'batch_size': kwargs.get('batch_size', 2),
        'buffer_size': kwargs.get('buffer_size', 10),
        'epochs': kwargs.get('epochs', None),
        'num_parallel_calls': kwargs.get('num_parallel_calls', 2),
        'img_sizes': kwargs.get('img_sizes'),
        'padding': kwargs.get('padding')
    }

    # Fetch the tfrecords
    records = get_all_files_containing(data_dir, input_type, 'tfrecords')

    return get_input_feeder(records, params)

def get_tf_record_image_size(data_dir):
    """Determine the image size in the tfrecords.

    Since Tensorflow wants static shape, it is not possible to infer the size
    of the images dynamically. We therefor read one sample of the record and
    infer the size of the image

    Parameters
    ----------
    records: list
        Full paths to tf records

    Returns
    -------
    shape: tuple
        The shape of the contained image
    """
    # Get a random tf record
    record = get_all_files_containing(data_dir, 'train', 'tfrecords')[0]
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tf.train.string_input_producer([record]))
    parsed = parse_single_example(serialized_example)

    params = {
        'batch_size': 1,
        'buffer_size': 2,
        'epochs': None,
        'num_parallel_calls': 2}
    ftr = (parsed['feature/height'], parsed['feature/width'], parsed['feature/channels'])
    lbl = (parsed['label/height'], parsed['label/width'], parsed['label/channels'])

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        ftr_val, lbl_val = sess.run([ftr, lbl])

        coord.request_stop()
        coord.join(threads)

    return ftr_val, lbl_val


if __name__ == "__main__":
    print(get_tf_record_image_size('/home/adamf/data/Karolinska/tfrecords_512'))
