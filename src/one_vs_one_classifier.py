import copy
from itertools import combinations
from tqdm import tqdm
import numpy as np


class OneVsOneClassifier:
    def __init__(self, binary_classifier):
        self.binary_classifier = binary_classifier
        self.classifiers = {} # dictionary to store the classifiers
        self.n_classes = None


    def fit(self, X, y):
        # Get all unique combinations of classes for one-vs-one
        class_combinations = list(combinations(set(y), 2))
        self.n_classes = np.max(y) + 1
        assert set(y) == set(range(self.n_classes)), "y should be a set of integers from 0 to n_classes - 1"

        # Train binary classifiers for each pair of classes
        for class_pair in tqdm(class_combinations, desc="Training binary classifiers"):
            class_0, class_1 = class_pair
            self.classifiers[class_pair] = copy.deepcopy(self.binary_classifier) # Deep copy
            X_binary = X[(y == class_0) | (y == class_1)]
            y_binary = y[(y == class_0) | (y == class_1)]
            y_binary = np.where(y_binary == class_1, 1, -1) # 1 for class 1, -1 for class 0
            self.classifiers[class_pair].fit(X_binary, y_binary)


    def predict(self, X, return_votes = False):
        # Make predictions using the trained binary classifiers
        predictions = np.zeros((len(X), self.n_classes), dtype=int)
        for class_pair, classifier in tqdm(self.classifiers.items(), desc="Making predictions"):
            pred_binary = classifier.predict(X)
            pred_binary_labels = np.where(pred_binary > 0, class_pair[1], class_pair[0])  # values in {class_pair[0], class_pair[1]}
            predictions[np.arange(len(X)), pred_binary_labels] += 1

        if return_votes:
            return predictions
        else:
            final_predictions = np.argmax(predictions, axis=1)
            return final_predictions
