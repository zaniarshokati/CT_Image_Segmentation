import os
import shutil

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from skimage import measure


def create_directory(path):
    """
    Create a directory at the specified path. If it already exists, delete it and recreate it.

    Args:
        path (str): The directory path to be created.

    Returns:
        None
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def create_mask_from_polygon(image, contours):
    """
    Creates a binary mask with the dimensions of the image by converting a list of polygon-contours to binary masks
    and merging them together.

    Args:
        image (numpy.ndarray): The image that the contours refer to.
        contours (list): List of contours.

    Returns:
        numpy.ndarray: Binary mask.
    """
    
    lung_mask = np.array(Image.new('L', image.shape, 0))
    for contour in contours:
        x, y = contour[:, 0], contour[:, 1]
        polygon_tuple = list(zip(x, y))
        img = Image.new('L', image.shape, 0)
        ImageDraw.Draw(img).polygon(polygon_tuple, outline=0, fill=1)
        mask = np.array(img)
        lung_mask += mask

    lung_mask[lung_mask > 1] = 1  # sanity check to make 100% sure that the mask is binary

    return lung_mask.T  # transpose it to be aligned with the image dims

def segment_intensity(ct_numpy, lower_bound=-1000, upper_bound=-300, threshold=0.95):
    """
    Segment regions in a CT image based on intensity values.

    Args:
        ct_numpy (numpy.ndarray): Input CT image as a NumPy array.
        lower_bound (int): Lower intensity threshold (default: -1000).
        upper_bound (int): Upper intensity threshold (default: -300).
        threshold (float): Contour detection threshold (default: 0.95).

    Returns:
        list: List of contours representing segmented regions.
    """
    # Clip the CT image to the specified intensity range
    clipped_image = clip_and_binarize_ct(ct_numpy, lower_bound, upper_bound)

    # Find contours in the clipped image
    contours = measure.find_contours(clipped_image, threshold)

    return contours

def is_closed_contour(contour):
    """
    Check if a contour is closed.

    Args:
        contour (numpy.ndarray): Contour to check.

    Returns:
        bool: True if the contour is closed, False otherwise.
    """
    return np.all(contour[0] == contour[-1])

def euclidean_dist(dx, dy):
    """
    Calculate the Euclidean distance between two points in 2D space.

    Args:
        dx (float): The horizontal distance.
        dy (float): The vertical distance.

    Returns:
        float: The Euclidean distance.
    """
    return np.sqrt(dx**2 + dy**2)

def find_lung_contours(contours, min_volume=2000):
    """
    Identifies and returns the contours corresponding to the lung area.

    Args:
        contours (list): List of detected contours.
        min_volume (float): Minimum volume threshold for a contour to be considered (default: 2000).

    Returns:
        list: Contours corresponding to the lung area.
    """

    selected_contours = [(contour, ConvexHull(contour).volume) for contour in contours if ConvexHull(contour).volume > min_volume and is_closed_contour(contour)]

    if len(selected_contours) > 2:
        selected_contours.sort(key=lambda x: x[1])
        lung_contours = [contour for contour, _ in selected_contours[:-1]]
        return lung_contours
    
    lung_contours = [contour for contour, _ in selected_contours]
    return lung_contours

def display_contours(image, contours, title=None, save=False):
    """
    Display an image with overlaid contours.

    Args:
        image (numpy.ndarray): The image to display.
        contours (list): List of contours to overlay on the image.
        title (str): Title for the displayed image (default: None).
        save (bool): If True, save the image to a file; otherwise, display it (default: False).

    Returns:
        None
    """
    fig, ax = plt.subplots()
    ax.imshow(image.T, cmap='gray')

    for contour in contours:
        x, y = contour[:, 0], contour[:, 1]
        ax.plot(x, y, linewidth=1)

    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title)

    if save:
        plt.savefig(title)
        plt.close(fig)
    else:
        plt.show()

def show_image_slice(image_slice):
    """
    Display a 2D image slice.

    Args:
        image_slice (numpy.ndarray): 2D array representing the image slice.

    Returns:
        None
    """
    plt.figure()
    plt.imshow(image_slice.T, cmap="gray", origin="lower")


def overlay_image_with_mask(image, mask):
    """
    Overlay an image with a mask for visualization.

    Args:
        image (numpy.ndarray): The grayscale image.
        mask (numpy.ndarray): The mask to overlay.

    Returns:
        None
    """
    plt.figure()
    plt.imshow(image.T, cmap='gray', interpolation='none')
    plt.imshow(mask.T, cmap='jet', interpolation='none', alpha=0.5)

def save_nifty_binary_mask(binary_mask, output_name, affine_matrix):
    """
    Save a binary mask as a NIfTI file, converting values to 255 for display in NIfTI viewer.

    Args:
        binary_mask (numpy.ndarray): The binary mask.
        output_name (str): The name for the output NIfTI file.
        affine_matrix (numpy.ndarray): The 4x4 affine transformation matrix.

    Returns:
        None
    """
    # Convert binary mask values to 255
    binary_mask[binary_mask == 1] = 255

    # Create a NIfTI image
    nifti_image = nib.Nifti1Image(binary_mask, affine_matrix)

    # Save the NIfTI image as a compressed (.nii.gz) file
    nib.save(nifti_image, output_name + '.nii.gz')

def extract_pixel_dimensions(ct_img):
    """
    Extract the pixel dimensions of a CT image.
    This function retrieves the pixel dimensions for the X and Y axes.
    
    Args:
        ct_img: nib image

    Returns:
        List of the 2 pixel dimensions [pixdimX, pixdimY]
    """
    header = ct_img.header
    pix_dim = header["pixdim"]
    dim = header["dim"]

    # Find the indices of the two largest dimensions
    max_indices = np.argpartition(dim, -2)[-2:]

    # Extract the pixel dimensions for the largest and second largest dimensions
    pixdimX = pix_dim[max_indices[0]]
    pixdimY = pix_dim[max_indices[1]]

    return [pixdimX, pixdimY]

def clip_and_binarize_ct(ct_numpy, lower_bound, upper_bound):
    """
    Clips CT values to a predefined range and binarizes them.

    Args:
        ct_numpy (numpy.ndarray): Input CT image as a NumPy array.
        lower_bound (float): Lower intensity threshold for clipping.
        upper_bound (float): Upper intensity threshold for clipping.

    Returns:
        numpy.ndarray: Binarized CT image.
    """
    # Clip the CT values to the specified range
    clipped_image = np.clip(ct_numpy, lower_bound, upper_bound)

    # Binarize the clipped image
    binarized_image = np.where(clipped_image == upper_bound, 1, 0)

    return binarized_image

def compute_lung_area(binary_mask, pixel_dimensions):
    """
    Compute the area (in mm^2) of a binary mask using pixel dimensions.

    Args:
        binary_mask (numpy.ndarray): Binary lung mask.
        pixel_dimensions (tuple): Tuple with two values representing pixel dimensions (e.g., (pixel_size_x, pixel_size_y)).

    Returns:
        float: The lung area in mm^2.
    """
    # Ensure the binary mask contains only 0s and 1s
    binary_mask = np.clip(binary_mask, 0, 1)
    
    # Calculate the lung area by counting non-zero pixels and multiplying by pixel area
    lung_area = np.sum(binary_mask) * (pixel_dimensions[0] * pixel_dimensions[1])
    
    return lung_area

def denoise_vessels(lung_contours, vessels):
    """
    Denoise vessels by removing vessel pixels that are close to lung contours.

    Args:
        lung_contours (list): List of lung contours.
        vessels (numpy.ndarray): Binary vessel mask.

    Returns:
        numpy.ndarray: Denoised vessel mask.
    """
   # Get non-zero coordinates of vessels
    vessels_coords_x, vessels_coords_y = np.nonzero(vessels)

    # Define a threshold distance for denoising
    threshold_distance = 0.1

    for contour in lung_contours:
        x_points, y_points = contour[:, 0], contour[:, 1]
        for (coord_x, coord_y) in zip(vessels_coords_x, vessels_coords_y):
            # Calculate Euclidean distance between contour points and vessel coordinates
            distances = np.sqrt((x_points - coord_x) ** 2 + (y_points - coord_y) ** 2)
            
            # Check if any distance is less than the threshold, and set vessel pixel to 0
            if np.any(distances <= threshold_distance):
                vessels[coord_x, coord_y] = 0
    return vessels
