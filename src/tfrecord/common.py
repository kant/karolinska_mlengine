import tensorflow as tf


def _int64_feature(value):
    """Return value as a int64 feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Return value as a byte feature."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_img_img(tfrecord_path, ftr_chunks, lbl_chunks, wgt_chunks, get_tfexample):
    """Convert the lists given into tfrecords.

    Parameters
    ----------
    tfrecord_path: list
        list of tfrecord paths to use for conversions
    ftr_chunks: list
        List of lists of paths to feature images per tf_record
    lbl_chunks: list
        List of lists of paths to feature images per tf_record
    get_tfexample: method
        A method that takes in the feature and label and returns the tf example
        feature that is desired
    """
    with tf.Graph().as_default():
        with tf.Session('') as sess:
            n_shards = len(tfrecord_path)
            for i, (irecord, features, labels, weights) in \
                    enumerate(zip(tfrecord_path, ftr_chunks, lbl_chunks, wgt_chunks)):

                print("Shard %d/%d - %d total files" % (i + 1, n_shards, len(list(ftr_chunks))))
                with tf.python_io.TFRecordWriter(irecord) as tfrecord_writer:
                    for iftr, ilbl, iwgt in zip(features, labels, weights):
                        example = get_tfexample(iftr, ilbl, iwgt)
                        tfrecord_writer.write(example.SerializeToString())
