from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist

from dataset import DatasetCreator


class MNIST3(DatasetCreator):
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

        num_3_train = len(y_train[y_train == 3])
        num_3_test = len(y_test[y_test == 3])

        # total amount of 3 images = total amount of other images
        index_train = np.hstack((np.where([y_train != 3])[1][:num_3_train], np.where((y_train == 3))[0]))
        index_test = np.hstack((np.where([y_test != 3])[1][:num_3_test], np.where((y_test == 3))[0]))

        X_train = X_train[index_train]
        X_test = X_test[index_test]

        y_train = y_train[index_train]
        y_test = y_test[index_test]

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

        idx_train = np.arange(len(X[0]))
        idx_test = np.arange(len(X[1]))

        np.random.shuffle(idx_train)
        np.random.shuffle(idx_test)

        data = namedtuple("data", ["X", "y", "idx"])
        train = data(X[0][idx_train], y[0][idx_train], idx_train)
        valid = train
        test = data(X[1][idx_test], y[1][idx_test], idx_test)
        return train, valid, test

if __name__ == '__main__':
    X, y = MNIST3.get_data(filepath=Path(""))

    n_plot = 10
    fig, ax = plt.subplots(1, n_plot, figsize=(20, 4))
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(X[0][i], (28, 28))
        ax.imshow(plottable_image, cmap='gray_r')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
