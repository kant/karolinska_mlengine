import tensorflow as tf
import numpy as np
import os
from PIL import Image
import argparse

from src.lib.imgops import plot_image_from_array, image_to_base64_websafe_resize, load_image
from src.lib.fileops import get_all_files

def get_predictor(saved_model_dir):
    predictor_fn = tf.contrib.predictor.from_saved_model(
        export_dir=saved_model_dir,
        signature_def_key="prediction"
    )
    return predictor_fn

def image_to_websafe(image_path):
    input_list = {'input': [image_to_base64_websafe_resize(image_path, (512, 512))]}

    return input_list


def apply_tinge(org_path, prediction):
    # Load and make the predicted cells a bit more red
    img, _ = load_image(org_path, size=(512, 512))
    original = np.stack([img] * 3, 2)
    x_idx, y_idx = np.where(prediction)
    original[x_idx, y_idx, 2] /= 10

    return original


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prediction_model',
        help='Path to the prediction model (folder with .pb file)',
        required=True
    )
    parser.add_argument(
        '--image_folder',
        help='Path to folder containing images to segment',
        required=True
    )
    parser.add_argument(
        '--output_location',
        help='Path to save the output prediction',
        required=True
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_location):
        os.makedirs(args.output_location)

    all_images = get_all_files(args.image_folder)
    predict = get_predictor(args.prediction_model)
    n_images = len(all_images)
    for i, image in enumerate(all_images):
        safe = image_to_websafe(image)
        response = predict(safe)
        prediction = np.squeeze(response['classes'])
        img = Image.fromarray(apply_tinge(image, prediction))
        save_path = os.path.join(args.output_location, os.path.basename(image))
        img.save(save_path)
