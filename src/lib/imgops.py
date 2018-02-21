from PIL import Image, ImageFilter
import matplotlib
import base64
from io import BytesIO
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
import scipy.ndimage as ndimage
import cv2

from src.lib.fileops import get_all_files


def load_image(img_path, size=(), scale=None, interpolation=Image.ANTIALIAS):
    """Load an image as an array, perform resizing if desired.

    Parameters
    ----------
    img_path : str
        Path to the image in question
    size : tuple, optional
        Desired output size of the image
    scale : float/None, optional
        Desired rescaling of the image
    interpolation : ANTIALIAS, optional
        Desired interpolation to use for resizing

    Returns
    -------
    np.array
        The image in numpy array format
    np.array
        The shape of said image
    """
    img = Image.open(img_path)
    if scale:
        image_size = img.size
        size = tuple([int(dim * scale) for dim in image_size])

    if size:
        img = img.resize(size, Image.ANTIALIAS)
    arr = np.array(img)

    arr_shape = arr.shape
    if len(arr.shape) == 2:
        arr_shape = (arr_shape[0], arr_shape[1], 1)
    return arr, arr_shape

def plot_image_from_array(img_arr, show_type="PIL"):
    """Plot an image from a numpy array.

    If you are running from inside a docker, use matplotlib, it's more stable.

    Parameters
    ----------
    img_arr : np.ndarray
        Numpy array
    show_type : str, optional
        Decide on which method to use for plotting

    """
    assert show_type in ["PIL", "matplotlib"]
    if img_arr.max() <= 1.0:
        img_arr = img_arr.astype(np.uint8)*255
    if show_type == "PIL":
        img = Image.fromarray(img_arr)
        img.show()
    else:
        plt.imshow(img_arr, interpolation='nearest')
        plt.show()


def image_to_base64_websafe_resize(image_path, size):
    """Converts an image into a websafe base64 format.

    Parameters
    ----------
    image_path : TYPE
        Path to image in question
    size : TYPE
        Desired size to convert the iamge to

    Returns
    -------
    base64
        Web safe base64 format

    """
    img = Image.open(image_path, mode='r')
    img = img.resize(size, Image.ANTIALIAS)
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')

    image_content = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    # Websafe means that + and / are eplaced with - and _
    image_content = image_content.replace('+', '-')
    image_content = image_content.replace('/', '_')

    return image_content


def create_weight_map(mask_path, effective_pix=100, sigma=0.05, weight=15):
    """Create an exponential weighting mask for a mask image.

    Parameters
    ----------
    mask_path : str
        Path to image
    effective_pix : int, optional
        How far away the exponent edge should extend
    sigma : float, optional
        How fast the exponential should decay
    weight : int, optional
        Scaling of exponent

    Returns
    -------
    np.ndarray, uint8
        Numpy array with exponential values instead of the original mask, scaled to 255

    """
    image_org = Image.open(mask_path).convert('RGB')
    image_boundary = image_org.filter(ImageFilter.FIND_EDGES)
    image_boundary = np.array(image_boundary)[:, :, 0] == 255
    image_mask = np.array(image_org)[:, :, 0] == 255

    # Create the sweeping distance matrix
    x_max, y_max = image_mask.shape
    x_effective = np.arange(-effective_pix, effective_pix)
    y_effective = x_effective
    d_effective = np.sqrt(x_effective**2 + y_effective**2)
    d_in_range = d_effective <= effective_pix
    x_idx, y_idx = (x_effective[d_in_range], y_effective[d_in_range])
    x, y = np.meshgrid(x_idx, y_idx)
    d = np.sqrt(x**2 + y**2)
    sweep_weight = weight*np.exp(-d**2 / 2*sigma**2)

    weight_map = np.ones(image_boundary.shape, dtype=np.float64)
    for xi, yi in zip(*np.where(image_boundary)):
        x_here = xi + x
        y_here = yi + y
        valid_x = np.logical_and(x_here >= 0, x_here < x_max)
        valid_y = np.logical_and(y_here >= 0, y_here < y_max)
        valid_both = np.logical_and(valid_x, valid_y)
        weight_map[x_here[valid_both], y_here[valid_both]] += sweep_weight[valid_both]

    weight_map = np.divide(weight_map, weight_map.max())
    weight_map[image_mask] = 1
    weight_map *= 255

    return weight_map.astype(np.uint8)

def process_one(paths):
    """Process an mask image, and save the exponent weighting image.

    Parameters
    ----------
    paths : list
        First item is the mask_image, and second is the output path to save

    Returns
    -------
    None

    """
    image_path, output_path = paths
    print("Processing %s" % os.path.basename(image_path))
    if os.path.isfile(output_path):
        print("\tAlready exists..")
        return
    weight_map = create_weight_map(image_path)
    img = Image.fromarray(weight_map)
    img.save(output_path)
    print("\tDone..")
    return


def create_weights_from_masks(mask_folder, output_folder, NUM_PROCS=8):
    """Creates weight image for all masks in mask_folder and saves to output folder.

    This process uses paralell processing to speed this up.

    Parameters
    ----------
    mask_folder : str
        Path to mask folder
    output_folder : str
        Desired output folder
    NUM_PROCS : int, optional
        How many threads to spawn

    """
    from multiprocessing import Pool
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    masks = get_all_files(mask_folder)
    input_pool = []
    for iMask in masks:
        image_path = os.path.basename(iMask)
        output_path = os.path.join(output_folder, image_path)
        input_pool.append([iMask, output_path])

    pool = Pool(processes=NUM_PROCS)
    pool.map(process_one, input_pool)

def get_weight_array(image_path, w0=10, sigma=8):

    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Calculate the contours
    img_empty = 255 * np.ones(img.shape, dtype=np.uint8)
    _, contours, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw them onto the rgb image
    all_contours_together = cv2.drawContours(img_empty.copy(), contours, -1, (0, 0), -1)

    # Create an empty container to fill in
    distance_map = np.zeros(list(img.shape) + [len(contours)])

    # Loop through all contours
    for i, i_contour in enumerate(contours):
        this_contour = cv2.drawContours(img_empty.copy(), [i_contour], -1, (0, 0), -1)
        edt = ndimage.distance_transform_edt(this_contour == 255)
        distance_map[:, :, i] = edt

    # distance_map = np.stack(collected_edt, -1)
    largest_distance = distance_map.max() + 1

    # Anywhere where there is zero, we set to the largest number
    distance_map[distance_map == 0] = largest_distance

    # Calculate closest and next closest cell to all pixels
    col_idx, row_idx = [
        val.transpose() for val in np.meshgrid(*[np.arange(dim) for dim in img.shape])
    ]

    cell_idx = np.argmin(distance_map, -1)
    d1_val = distance_map[col_idx, row_idx, cell_idx]
    distance_map[col_idx, row_idx, cell_idx] = largest_distance + 1
    d2_val = np.min(distance_map, -1)

    # Class weights
    n_total_pixels = img.shape[0] * img.shape[1]
    n_background_pixels = np.sum(all_contours_together != 0)
    n_cell_pixels = n_total_pixels - n_background_pixels
    bg_ratio = 1 - float(n_cell_pixels) / n_total_pixels
    cell_ratio = 1 - bg_ratio

    # Make sure the ratios down "explode" if there are only few cells in image
    ratios = np.array([bg_ratio / cell_ratio, 1.0])
    ratios = np.divide(ratios, ratios.max())

    class_weights = np.ones(img.shape, dtype=np.float32)
    class_weights[all_contours_together == 0] = ratios[0]
    class_weights[all_contours_together == 255] = ratios[1]

    # Exponential distance weights
    weights = w0 * np.exp(-np.divide(np.power(d1_val + d2_val, 2), 2 * sigma**2))

    # Make sure insides of cells are still 0
    weights[all_contours_together == 0] = 0

    return weights + class_weights

def create_weight_file(image_path, output_folder):
    weights = get_weight_array(image_path, w0=10, sigma=25)

    # Theoretical maximum weight
    w_max = 10 + 1

    # Normalize to 255
    weights = np.divide(255 * weights, w_max).astype(np.uint8)
    file_name = os.path.basename(image_path).split('.')[0]
    save_path = os.path.join(output_folder, "%s.png" % file_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img = Image.fromarray(weights)
    img.save(save_path)


def plot_sample_from_tf_record(data_dir):
    """Plot a sample from the tfrecord for debugging."""

    from src.lib.fileops import load_dataset_config
    record = get_all_files_containing(data_dir, 'train', 'tfrecord')
    shape = get_tf_record_image_size(data_dir)

    image, label = input_function(
        filenames=record,
        network_type='segmentation',
        img_sizes=shape,
        batch_size=5,
        epochs=None,
        buffer_size=2,
        num_parallel_calls=1)()

    with tf.Session() as sess:
        print("Coord and queue")
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)
        img_val, lbl_val, mask = sess.run([image, label, mask_op])
        tf.summary.image('mask', mask_op)

    # Now lets de-normalize and plot
    image = ((img_val['image'][0, ::] + 0.5) * 255).astype(np.uint8)
    # mask = lbl_val['mask'][0, ::] * 255
    # gradient = lbl_val['gradient'][0, ::] * 255
    # energy = lbl_val['energy'][0, ::] * 255

    from src.lib.imgops import plot_image_from_array
    plot_image_from_array([image, mask[0, ::]])


def edt_from_mask(mask_image):
    """Given a mask, create the directional gradients and energy map.

    The energy map is then binned into 16 distinct bins. The interval of each
    bin is taken from the original authors implementation, where he does not
    comment in detail why those were chosen.

    boundaries = [0,1,2,3,4,5,7,9,12,15,19,24,30,37,45,54,Inf];

    Parameters
    ----------
    mask_image : str or numpy array
        Path to image, or an boolean numpy array of size (W, H)
    pixel_boundary : TYPE, optional
        Description

    Returns
    -------
    Numpy array
        (W, H, 2) gradient direction mask

    Deleted Parameters
    ------------------
    debug_plot : bool, optional
        Plot the results or not
    """
    # The first bin MUST start with 0
    image_etc = ndimage.distance_transform_edt(mask_image)

    # Scale with shape
    PIXEL_BOUNDARY = [0, 1, 2, 3, 4, 5, 7, 9, 12, 14, 16, 18, 20, 24, 28]
    ratio = int(float(mask_image.shape[0]) / 512.0)
    pixel_boundary = [int(val * ratio) for val in PIXEL_BOUNDARY]
    assert pixel_boundary[0] == 0

    # Calculate the gradient
    image_grad_y, image_grad_x = np.gradient(image_etc)
    image_grad_norm = np.sqrt(image_grad_x**2 + image_grad_y**2)

    # Ignore divide by zero warning
    with np.errstate(divide='ignore', invalid='ignore'):
        image_grad_y = np.divide(image_grad_y, image_grad_norm)
        image_grad_x = np.divide(image_grad_x, image_grad_norm)

    # Fix for zero gradients (the boundary)
    image_grad_y[image_grad_norm == 0] = 0
    image_grad_x[image_grad_norm == 0] = 0

    # Get the one-hot energy, but as integers, not seperate layers.
    # We need to set the background as -1 before doing anything else, so that it gets put into the
    # <-inf, 0> bin
    image_etc_new = image_etc.copy()
    image_etc_new[image_etc_new == 0] = -1
    one_hot = np.digitize(image_etc_new, pixel_boundary).astype(np.uint8)

    return np.stack([image_grad_x, image_grad_y], -1), one_hot


def get_gradient_and_energy(loaded_label):

    # Get unique instances
    idx_instance = np.unique(loaded_label)

    # Remove 0
    idx_instance = np.delete(idx_instance, np.where(idx_instance == 0)[0])

    # Pre-allocated size
    shape = [len(idx_instance)] + list(loaded_label.shape)

    # Now, we need to create the gradient maps and watershed energy
    # for each instance mask.
    energy_map = np.zeros(shape=shape, dtype=np.float64)
    gradient_map = np.stack([energy_map.copy(), energy_map.copy()], -1)

    for idx, i_ener, i_grad in zip(idx_instance, energy_map, gradient_map):
        gradient, energy = edt_from_mask(loaded_label == idx)
        i_ener += energy
        i_grad += gradient

    # Now merge all instances
    merged_gradient = np.sum(gradient_map, axis=0)
    merged_energy = np.sum(energy_map, axis=0)

    # We should have at most 16 different values, the max is then 16-1 = 15
    assert merged_energy.max() <= 15

    return merged_energy, merged_gradient


def process_single_label(loaded_label):
    merged_energy, merged_gradient = get_gradient_and_energy(loaded_label)
    return merged_energy.astype(np.uint8)


if __name__ == "__main__":
    plot_sample_from_tf_record('')