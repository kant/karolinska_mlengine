import tensorflow as tf
import numpy as np
import os
from PIL import Image
import argparse
import cv2

from src.lib.imgops import plot_image_from_array, array_to_base64_websafe_resize, \
    load_image, image_to_base64_websafe_resize
from src.lib.fileops import get_all_files

def get_predictor(saved_model_dir):
    predictor_fn = tf.contrib.predictor.from_saved_model(
        export_dir=saved_model_dir,
        signature_def_key="prediction"
    )
    return predictor_fn

def image_to_websafe_watershed(image_path, mask_array):
    img_array, _ = load_image(image_path, size=(512, 512))
    total_array = np.stack([img_array, mask_array], 2).astype(np.uint8)
    input_list = {
        'input': [array_to_base64_websafe_resize(total_array)]}

    return input_list

def image_to_websafe_mask(image_path):
    input_list = {'input': [image_to_base64_websafe_resize(image_path, (512, 512))]}

    return input_list


def apply_tinge(org_path, prediction, threshold=1):
    # Load and make the predicted cells a bit more red
    img, _ = load_image(org_path, size=(512, 512))
    gain = np.floor(255 / prediction.max())
    org = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    pred = prediction > threshold
    pred = np.stack([pred.astype(np.uint8)] * 3, 2)
    pred = cv2.cvtColor(255*pred, cv2.COLOR_RGB2GRAY)
    _, contours, _ = cv2.findContours(pred.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw them onto the rgb image
    org = cv2.drawContours(org.copy(), contours, -1, (255, 0), 0)

    return org


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mask_model',
        help='Path to the mask prediction model (folder with .pb file)',
        required=True
    )
    parser.add_argument(
        '--watershed_model',
        help='Path to the watershed prediction model (folder with .pb file)',
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
    parser.add_argument(
        '--energy_threshold',
        help='Which energy contour to plot',
        default=0,
        type=int
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_location):
        os.makedirs(args.output_location)

    all_images = get_all_files(args.image_folder)
    mask_predictor = get_predictor(args.mask_model)
    watershed_predictor = get_predictor(args.watershed_model)
    n_images = len(all_images)
    for i, image in enumerate(all_images):
        # Get a mask prediction
        safe = image_to_websafe_mask(image)
        mask_response = mask_predictor(safe)
        predicted_mask = np.squeeze(mask_response['classes'])

        # Pipe to watershed prediction
        safe = image_to_websafe_watershed(image, predicted_mask)
        mask_response = watershed_predictor(safe)
        predicted_energy = np.squeeze(mask_response['classes'])

        img = Image.fromarray(apply_tinge(image, predicted_energy, args.energy_threshold))
        save_path = os.path.join(args.output_location, os.path.basename(image))
        img.save(save_path)
