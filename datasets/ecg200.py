import sys
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff

from dataset import DatasetCreator

sys.path.insert(0, '..')
from collections import namedtuple

class ECG200(DatasetCreator):
    """
    Subclass of DatasetCreator to specify dataset selection. None of the abstract functions from abstract class can
    be overwritten.
    """

    @staticmethod
    def get_data(filepath) -> tuple:
        """
        return artificial data
        """
        def read_arff(file_path, encoding: str) -> tuple:
            """
            reading of arrf files
            :param file_path: path to file
            :return: data array with data of selected events
            """
            with open(file_path, 'rt', encoding=encoding) as f:
                data_read = f.read()
            stream = StringIO(data_read)
            return arff.loadarff(stream)

        data_train = read_arff(filepath / Path("ECG200") / "ECG200_TRAIN.arff", encoding="utf-8")
        data_test = read_arff(filepath / Path("ECG200") / "ECG200_TEST.arff", encoding="utf-8")

        data_train_list = data_train[0]
        data_train_array = np.empty(shape=(len(data_train_list), (len(data_train_list[0]))))
        for index, signal in enumerate(data_train_list):
            data_train_array[index, :] = list(signal)

        data_test_list = data_test[0]
        data_test_array = np.empty(shape=(len(data_test_list), (len(data_test_list[0]))))
        for index, signal in enumerate(data_test_list):
            data_test_array[index, :] = list(signal)

        data_combined = np.concatenate([data_train_array, data_test_array])

        #data_combined = np.stack((data_train, data_train))

        y = data_combined[:, -1]
        X = data_combined[:, :-1, np.newaxis]

        return X, y

    @staticmethod
    def split_data(X, y) -> tuple:
        """
        abstract method to select events for dataset
        """
        idx = np.arange(len(X))
        data = namedtuple("data", ["X", "y", "idx"])

        train = data(X[:100], y[:100], idx[:100])
        valid = train
        test = data(X[100:], y[100:], idx[100:])

        return train, valid, test

if __name__ == '__main__':
    X, y = ECG200.get_data(filepath=Path(""))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(X[y == -1][:, :, 0].T, alpha=0.3)
    ax[0].set_title("class 0")
    ax[1].plot(X[y == 1][:, :, 0].T, alpha=0.3)
    ax[1].set_title("class 1")
    plt.show()