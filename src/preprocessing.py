import numpy as np
from skimage.feature import hog
from skimage import exposure
from tqdm import tqdm
from src.utils import row_to_image, data_to_list_of_images



def extract_hog_features(images, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extract HOG features from a list of images.

    Parameters:
    - images: List of 3D NumPy arrays representing color images. Each array should have shape (3, 32, 32).
    - orientations: Number of gradient orientations.
    - pixels_per_cell: Size (in pixels) of a cell.
    - cells_per_block: Number of cells in each block.
    - multichannel: Whether the input images are multichannel (color).

    Returns:
    - hog_features: List of HOG feature vectors for each input image.
    """
    hog_features = []

    for image in tqdm(images, "preprocessing"):
        # Transpose the image to have channels as the last dimension
        # Extract HOG features
        features, _ = hog(image, orientations=orientations,
                          pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                          visualize=True,channel_axis = -1)

        # Perform global contrast normalization on the HOG features
        features = exposure.rescale_intensity(features, in_range=(0, 10))

        # Append the features to the list
        hog_features.append(features)

    return hog_features



def preprocess_image_HOG(X, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), normalize = True):
    # X is a 4D array of shape (n_samples, 32, 32, 3)
    X_preprocessed = extract_hog_features(X, orientations, pixels_per_cell, cells_per_block)
    X_preprocessed = np.stack(X_preprocessed, axis=0)
    return X_preprocessed