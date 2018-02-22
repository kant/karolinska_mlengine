import tensorflow as tf
import numpy as np
import os
from PIL import Image
import argparse

from src.lib.imgops import plot_image_from_array, array_to_base64_websafe_resize, load_image
from src.lib.fileops import get_all_files

def get_predictor(saved_model_dir):
    predictor_fn = tf.contrib.predictor.from_saved_model(
        export_dir=saved_model_dir,
        signature_def_key="prediction"
    )
    return predictor_fn

def image_to_websafe(image_path, label_path):
    img_array, _ = load_image(image_path, size=(512, 512))
    mask_array, _ = load_image(label_path, size=(512, 512))
    total_array = np.stack([img_array, mask_array], 2).astype(np.uint8)
    input_list = {
        'input': [array_to_base64_websafe_resize(total_array)]}

    return input_list


def apply_tinge(org_path, prediction):
    # Load and make the predicted cells a bit more red
    img, _ = load_image(org_path, size=(512, 512))
    original = np.stack([img] * 3, 2)
    original[:, :, 0] = 0
    original[:, :, 0] = prediction*15

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
        '--mask_folder',
        help='Path to folder containing segment masks (binary)',
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
    all_masks = get_all_files(args.mask_folder)
    predict = get_predictor(args.prediction_model)
    n_images = len(all_images)
    for i, (image, mask) in enumerate(zip(all_images, all_masks)):
        safe = image_to_websafe(image, mask)
        response = predict(safe)
        predicted_energy = np.squeeze(response['classes'])
        img = Image.fromarray(apply_tinge(image, predicted_energy))
        save_path = os.path.join(args.output_location, os.path.basename(image))
        img.save(save_path)
