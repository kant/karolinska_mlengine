import tensorflow as tf
import random
import numpy as np
import argparse
import os
from math import ceil
import src.tfrecord.common as common
from src.lib import fileops, imgops, listops

def _get_img_img_example(feature_path, label_path, weight_path, size=(), scale=None):
    """Load an image and label pair and return it as a training Example.

    Parameters
    ----------
    feature_path : str
        Path to the input image
    label_path : str
        Path to the label image
    weight_path : str
        Path to the weight image
    size : tuple, optional
        Desired output size of the images
    scale : None, optional
        desired rescaling of the images

    Returns
    -------
    tf.train.Example
        A tf.train.Example that is encoded into a tfrecord file
    """
    ftr, ftr_shape = imgops.load_image(feature_path, size=size, scale=scale)
    lbl, lbl_shape = imgops.load_image(label_path, size=size, scale=scale)
    wgt, wgt_shape = imgops.load_image(weight_path, size=size, scale=scale)
    erg = imgops.process_single_label(lbl)

    # Convert our instance label array into a binary mask
    lbl = (lbl > 0).astype(np.uint8)

    feature = {
        'feature/img': common._bytes_feature(tf.compat.as_bytes(ftr.tostring())),
        'feature/height': common._int64_feature(ftr_shape[0]),
        'feature/width': common._int64_feature(ftr_shape[1]),
        'feature/channels': common._int64_feature(ftr_shape[2]),
        'label/img': common._bytes_feature(tf.compat.as_bytes(lbl.tostring())),
        'label/height': common._int64_feature(lbl_shape[0]),
        'label/width': common._int64_feature(lbl_shape[1]),
        'label/channels': common._int64_feature(lbl_shape[2]),
        'weight/img': common._bytes_feature(tf.compat.as_bytes(wgt.tostring())),
        'weight/height': common._int64_feature(wgt_shape[0]),
        'weight/width': common._int64_feature(wgt_shape[1]),
        'weight/channels': common._int64_feature(wgt_shape[2]),
        'energy/img': common._bytes_feature(tf.compat.as_bytes(erg.tostring())),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def _img_img_read_and_decode(record):
    """Decdoes the above record into the desired compnents.

    Parameters
    ----------
    record : Serialized tfrecord
        A loaded tfrecord

    Returns
    -------
    tensor
        (W, H, C) Normalized input image of the some width, height, channel
    tensor
        (W, H, C) Label image of the some width, height, channel
    """

    feature = {
        'feature/img': tf.FixedLenFeature([], tf.string),
        'feature/height': tf.FixedLenFeature([], tf.int64),
        'feature/width': tf.FixedLenFeature([], tf.int64),
        'feature/channels': tf.FixedLenFeature([], tf.int64),
        'label/img': tf.FixedLenFeature([], tf.string),
        'label/height': tf.FixedLenFeature([], tf.int64),
        'label/width': tf.FixedLenFeature([], tf.int64),
        'label/channels': tf.FixedLenFeature([], tf.int64),
        'weight/height': tf.FixedLenFeature([], tf.int64),
        'weight/width': tf.FixedLenFeature([], tf.int64),
        'weight/channels': tf.FixedLenFeature([], tf.int64)
    }

    parsed = tf.parse_single_example(record, feature)

    # Get the sizes
    ftr_height = tf.cast(parsed['feature/height'], tf.int32)
    ftr_width = tf.cast(parsed['feature/width'], tf.int32)
    ftr_channel = tf.cast(parsed['feature/channels'], tf.int32)
    lbl_height = tf.cast(parsed['label/height'], tf.int32)
    lbl_width = tf.cast(parsed['label/width'], tf.int32)
    lbl_channel = tf.cast(parsed['label/channels'], tf.int32)

    # shape of image and annotation
    ftr_shape = tf.stack([ftr_height, ftr_width, ftr_channel])
    lbl_shape = tf.stack([lbl_height, lbl_width, lbl_channel])
    wgt_shape = tf.stack([wgt_height, wgt_width, wgt_channel])

    # read, decode and normalize image
    feature_img = tf.decode_raw(parsed['feature/img'], tf.uint8)
    feature_img = tf.cast(feature_img, tf.float32) * (1. / 255) - 0.5
    feature_img = tf.reshape(feature_img, [320, 479, 3])
    label_img = tf.decode_raw(parsed['label/img'], tf.uint8)
    label_img = tf.cast(label_img, tf.int32)
    label_img = tf.reshape(label_img, [320, 479, 1])

    return feature_img, label_img

class ImgImgParser():
    """This class simplifies creating TFRecords and providing input functions to an Estimator.

    Methods
    -------

    create_records
    get_input_feeder

    """
    def __init__(self,
                 output_path,
                 feature_folder,
                 label_folder,
                 weight_folder,
                 split=[0.8, 0.1, 0.1],
                 img_size=(),
                 img_scale=None,
                 randomize=True):
        """Initializes the class

        Parameters
        ----------
        output_path : str
            The output path to save any generated tfrecord files to
        feature_folder : str
            Path to the directory containgin all the input images
        label_folder : str
            Path to the directory containing all the label iamges
        split=[0.8, 0.1, 0.1] : list, optional
            The desired train, eval, test split to use
        img_scale : tuple, optional
            Desired output size of the images
        randomize=False : bool, optional
            Randomize the found files in the directories
        """
        self._feature_folder = feature_folder
        self._label_folder = label_folder
        self._weight_folder = weight_folder
        self._split = split
        self._output_path = output_path
        self._img_size = img_size
        self._img_scale = img_scale
        self._randomize = randomize

    def _get_image_paths(self, path):
        """Fetches all files in a given path."""
        return fileops.get_all_files(path)

    def create_records(self, shards=1):
        """Create n number of shards tfrecords at output_path.

        Parameters
        ----------
        shards : int, optional
            How many shards to split the data into
        """
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)

        feature_path = self._get_image_paths(self._feature_folder)
        label_path = self._get_image_paths(self._label_folder)
        weight_path = self._get_image_paths(self._weight_folder)

        if self._randomize:
            c = list(zip(feature_path, label_path, weight_path))
            random.shuffle(c)
            feature_path, label_path, weight_path = zip(*c)

        # Get the data split
        split = fileops.split_lists([feature_path, label_path, weight_path], *self._split)
        split_names = ['train', 'val', 'test']

        # Try to split up the "shards" between the dataset splits
        test_shards = ceil(shards * self._split[2])
        val_shards = ceil(shards * self._split[1])
        train_shards = ceil(shards * self._split[0])
        n_shards = [train_shards, val_shards, test_shards]

        # Wrap the example generator
        def example_gen(feature, label, weight):
            return _get_img_img_example(feature, label, weight, size=self._img_size, scale=self._img_scale)

        # Loop through
        for _name, paths, _n_shard in zip(split_names, split, n_shards):
            _n_shard = int(_n_shard)
            feature_paths, label_paths, weight_paths = paths

            # Create the tfrecord name for each shard
            tf_names = [os.path.join(self._output_path,
                                     '%s_shard_%d.tfrecords' % (_name, i))
                        for i in range(_n_shard)]
            sharded_features = listops.chunker_list(feature_paths, _n_shard)
            sharded_labels = listops.chunker_list(label_paths, _n_shard)
            sharded_weights = listops.chunker_list(weight_paths, _n_shard)

            common.convert_img_img(tf_names, sharded_features, sharded_labels, sharded_weights, example_gen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_path',
        help='Specify the output dir of the tfrecords',
        required=True,
        type=str
    )
    parser.add_argument(
        '--feature_folder',
        help='Path to the folder containing all the images',
        required=True,
        type=str
    )
    parser.add_argument(
        '--label_folder',
        help='Path to the folder containing the mask images',
        required=True,
        type=str
    )
    parser.add_argument(
        '--weight_folder',
        help='Path to the folder containing the weights of mask images',
        required=True,
        type=str
    )
    parser.add_argument(
        '--split',
        nargs=3,
        help='The train/eval/test split, default os [0.8, 0.1, 0.1]',
        default=[0.8, 0.1, 0.1],
        type=float
    )
    parser.add_argument(
        '--img_size',
        nargs=2,
        help='Desired size of the images, default is no change (None)',
        default=[],
        type=int
    )
    parser.add_argument(
        '--img_scale',
        help='Desired scaling of the images, default is no change (None)',
        default=None,
        type=float
    )
    parser.add_argument(
        '--randomize',
        help='Controls if the found images should be randomized before tfrecord is created',
        default=True
    )
    parser.add_argument(
        '--shards',
        help='How many shards the tfrecords should be split up into (Bigger = More RAM needed)',
        default=10,
        type=int
    )

    args = parser.parse_args()
    arguments = args.__dict__
    shards = arguments.pop('shards')
    img_parser = ImgImgParser(**arguments)
    img_parser.create_records(shards)
