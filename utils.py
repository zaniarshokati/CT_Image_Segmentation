import os
import shutil

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from skimage import measure


def make_dirs(path):
    """
    Creates the directory as specified from the path
    in case it exists it deletes it
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
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


def contour_distance(contour):
    """
    Given a set of points that may describe a contour
     it calculates the distance between the first and the last point
     to infer if the set is closed.
    Args:
        contour: np array of x and y points

    Returns: euclidean distance of first and last point
    """
    dx = contour[0, 1] - contour[-1, 1]
    dy = contour[0, 0] - contour[-1, 0]
    return euclidean_dist(dx, dy)


def euclidean_dist(dx, dy):
    return np.sqrt(np.power(dx, 2) + np.power(dy, 2))


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


def show_slice(slice):
    """
    Function to display an image slice
    Input is a numpy 2D array
    """
    plt.figure()
    plt.imshow(slice.T, cmap="gray", origin="lower")


def overlay_plot(im, mask):
    plt.figure()
    plt.imshow(im.T, 'gray', interpolation='none')
    plt.imshow(mask.T, 'jet', interpolation='none', alpha=0.5)


def save_nifty(img_np, name, affine):
    """
    binary masks should be converted to 255 so it can be displayed in a nii viewer
    we pass the affine of the initial image to make sure it exits in the same
    image coordinate space
    Args:
        img_np: the binary mask
        name: output name
        affine: 4x4 np array
    Returns:
    """
    img_np[img_np == 1] = 255
    ni_img = nib.Nifti1Image(img_np, affine)
    nib.save(ni_img, name + '.nii.gz')


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

def compute_area(mask, pixdim):
    """
    Computes the area (number of pixels) of a binary mask and multiplies the pixels
    with the pixel dimension of the acquired CT image
    Args:
        lung_mask: binary lung mask
        pixdim: list or tuple with two values

    Returns: the lung area in mm^2
    """
    mask[mask >= 1] = 1
    lung_pixels = np.sum(mask)
    return lung_pixels * pixdim[0] * pixdim[1]


def denoise_vessels(lung_contour, vessels):
    vessels_coords_x, vessels_coords_y = np.nonzero(vessels)  # get non zero coordinates
    for contour in lung_contour:
        x_points, y_points = contour[:, 0], contour[:, 1]
        for (coord_x, coord_y) in zip(vessels_coords_x, vessels_coords_y):
            for (x, y) in zip(x_points, y_points):
                d = euclidean_dist(x - coord_x, y - coord_y)
                if d <= 0.1:
                    vessels[coord_x, coord_y] = 0
    return vessels
