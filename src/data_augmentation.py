import numpy as np
import cv2

def augment_data_opencv(images, labels):
    """
    X_train_val = np.array(
    pd.read_csv(DATA_FOLDER + "Xtr.csv", header=None, sep=",", usecols=range(3072))
    )

    y_train_val = np.array(
        pd.read_csv(DATA_FOLDER + "Ytr.csv", sep=",", usecols=[1])
    ).squeeze()


    X_list = data_to_list_of_images(X_train_val, normalize=False)

    augmented_images, augmented_labels = augment_data_opencv(X_list, y_train_val)
    """
    # Initialize lists to store augmented images and labels
    assert len(images) == len(labels)
    augmented_images = []
    augmented_labels = []

    for i in range(len(images)):
        image = images[i]
        label = labels[i]
        augmented_images.append(image)
        augmented_labels.append(label)

        # Flip horizontally
        flipped_img = cv2.flip(image, 1)
        augmented_images.append(flipped_img)
        augmented_labels.append(label)

        # # Crop image
        # crop_size = (24, 24)
        # top_left_x = 4
        # top_left_y = 4
        # cropped_img = image[top_left_y:top_left_y + crop_size[0], top_left_x:top_left_x + crop_size[1]]
        # cropped_img = cv2.resize(cropped_img, (32, 32))
        # augmented_images.append(cropped_img)
        # augmented_labels.append(label)


    # Convert the lists to numpy arrays
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    return augmented_images, augmented_labels
