# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

sns.set_theme()

# Personal code
from src.utils import get_accuracy, get_date_time_string, plot_one_image, row_to_image, get_submission, data_to_list_of_images
from src.kernels import RBF, Linear, Polynomial
from src.kernel_classifiers import SVM

DATA_FOLDER = "data/"



from src.train_test_split import stratified_train_test_split
from src.one_vs_one_classifier import OneVsOneClassifier
from src.preprocessing import preprocess_image_HOG
from src.data_augmentation import augment_data_opencv


# Load data
X_train_val = np.array(
    pd.read_csv(DATA_FOLDER + "Xtr.csv", header=None, sep=",", usecols=range(3072))
)
X_test = np.array(
    pd.read_csv(DATA_FOLDER + "Xte.csv", header=None, sep=",", usecols=range(3072))
)
y_train_val = np.array(
    pd.read_csv(DATA_FOLDER + "Ytr.csv", sep=",", usecols=[1])
).squeeze()

# Split data
X_train, X_val, y_train, y_val = stratified_train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_seed=42
)

# Transform data into images
X_train = data_to_list_of_images(X_train, normalize=False)
X_val = data_to_list_of_images(X_val, normalize=False)
X_test = data_to_list_of_images(X_test, normalize=False)

# Data augmentation
X_train, y_train = augment_data_opencv(X_train, y_train)

# Data preprocessing (centering + additing features)
X_train = preprocess_image_HOG(
    X_train, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), normalize = False
)
X_val = preprocess_image_HOG(
    X_val, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), normalize = False
)
X_test = preprocess_image_HOG(
    X_test, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), normalize = False
)

# Choose binary classifier
gamma = 100
kernel = RBF(gamma).kernel
binary_classifier = SVM(kernel=kernel, C=100)

# Transform it into a multiclass classifier
ovo_classifier = OneVsOneClassifier(binary_classifier)

# Fit the classifier
ovo_classifier.fit(X_train, y_train)

# Predict on validation
y_val_pred = ovo_classifier.predict(X_val, return_votes=False)
accuracy = get_accuracy(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Predict on test
y_test_pred = ovo_classifier.predict(X_test, return_votes = False)

# Save the predictions of the test in the submissions folder
submission = get_submission(y_test_pred, name = 'Yte')

