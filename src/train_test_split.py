import numpy as np

def stratified_train_test_split(X, y, test_size=0.2, random_seed=42):
    np.random.seed(random_seed)
    unique_classes = np.unique(y)

    # Initialize arrays to store train and test indices
    train_indices = np.array([], dtype=int)
    test_indices = np.array([], dtype=int)

    # Iterate over unique classes
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        np.random.shuffle(class_indices)
        num_test_samples = int(len(class_indices) * test_size)
        train_indices = np.concatenate((train_indices, class_indices[num_test_samples:]))
        test_indices = np.concatenate((test_indices, class_indices[:num_test_samples]))

    # Shuffle the train and test arrays for randomness
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # Perform the split
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test