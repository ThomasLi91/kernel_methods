from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_date_time_string():
    # Get current date and time
    now = datetime.now()

    # Extract year, month, day, hour, and minute
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute

    # Format as a string of integers (yyyymmddHHMM)
    date_time_string = f"{month:02d}{day:02d}_{hour:02d}{minute:02d}_"

    return date_time_string


def plot_one_image(row_image: np.array):
    assert len(row_image) == 3072
    # reshape the image
    row_image = row_image.reshape((3, 32 * 32))
    minimum = row_image.min(axis=1, keepdims=True)
    maximum = row_image.max(axis=1, keepdims=True)

    row_image = (row_image - minimum) / (maximum - minimum)

    row_image = row_image.reshape((3, 32, 32))
    row_image = row_image.transpose(1, 2, 0)

    # normalize the image
    # row_image = (row_image - row_image.min()) / (row_image.max() - row_image.min())

    # plot the image
    plt.imshow(row_image)
    plt.show()
    return row_image


def row_to_image(row_image, normalize=True):
    """
    Takes as input the flattenned image and returns the original image
    of shape (32, 32, 3) and with values in the range [0, 1]
    """
    assert len(row_image) == 3072
    # reshape the image
    row_image = row_image.reshape((3, 32 * 32))
    minimum = row_image.min(axis=1, keepdims=True)
    maximum = row_image.max(axis=1, keepdims=True)

    if normalize:
        row_image = (row_image - minimum) / (maximum - minimum)

    row_image = row_image.reshape((3, 32, 32))
    row_image = row_image.transpose(1, 2, 0) # (32, 32, 3)

    return row_image



def data_to_list_of_images(X, normalize=False):
    n_samples = len(X)
    list_of_images = []
    for i in range(n_samples):
        list_of_images.append(row_to_image(X[i], normalize=normalize))
    list_of_images = np.stack(list_of_images, axis=0)
    return list_of_images


def get_accuracy(y_true : np.array, y_pred : np.array):
    """
    Calculate the accuracy of a model.
    """
    accuracy = np.mean(y_true == y_pred)
    return accuracy



def get_submission(y_test_pred : np.array, name = None):
    y_test_pred = y_test_pred.astype(int)
    submission = {'Prediction' : y_test_pred} 
    submission = pd.DataFrame(submission) 
    submission.index += 1
    PATH = "submissions/"
    if name is None:
        PATH += get_date_time_string() + '.csv'
    else:
        PATH += get_date_time_string() + name + '.csv'
    submission.to_csv(PATH,index_label='Id')
    return submission