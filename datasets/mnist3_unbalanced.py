from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist

from dataset import DatasetCreator


class MNIST3_unbalanced(DatasetCreator):
    """
    Subclass of DatasetCreator to specify dataset selection. None of the abstract functions from abstract class can
    be overwritten.
    """

    @staticmethod
    def get_data(filepath) -> tuple:
        """
        return artificial data
        """
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train / 255.
        X_test = X_test / 255.
        y_train = y_train
        y_test = y_test

        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        y_train[y_train != 3] = 0
        y_test[y_test != 3] = 0

        count_train = pd.DataFrame(y_train).value_counts()
        count_test = pd.DataFrame(y_test).value_counts()

        X = [X_train, X_test]
        y = [y_train, y_test]
        return X, y

    @staticmethod
    def split_data(X, y) -> tuple:
        """
        abstract method to select events for dataset
        """
        data = namedtuple("data", ["X", "y", "idx"])
        train = data(X[0], y[0], np.arange(len(X[0])))
        valid = train
        test = data(X[1], y[1], np.arange(len(X[1])))
        return train, valid, test

if __name__ == '__main__':
    X, y = MNIST3_unbalanced.get_data(filepath=Path(""))

    n_plot = 10
    fig, ax = plt.subplots(1, n_plot, figsize=(20, 4))
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(X[0][i], (28, 28))
        ax.imshow(plottable_image, cmap='gray_r')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
