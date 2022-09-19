from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class DatasetCreator(ABC):
    """
    abstract class which acts as a template to create datasets
    """

    @staticmethod
    @abstractmethod
    def get_data(filepath) -> tuple:
        """
        abstract method to select events for dataset
        """

    @staticmethod
    @abstractmethod
    def split_data(X, y) -> tuple:
        """
        abstract method to select events for dataset
        """
        idx = np.arange(len(X))
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, idx, test_size=0.5)

        data = namedtuple("data", ["X", "y", "idx"])
        train = data(X_train, y_train, idx_train)
        valid = train
        test = data(X_test, y_test, idx_test)

        return train, valid, test


def load_dataset(creator: DatasetCreator, filepath) -> tuple:
    """
    function loads dataset using DatasetCreator as creator
    """
    X, y = creator.get_data(filepath=filepath)

    train, valid, test = creator.split_data(X, y)

    enc = OneHotEncoder(categories='auto').fit(train.y.reshape(-1, 1))
    train = train._replace(y=enc.transform(train.y.reshape(-1, 1)).toarray())
    valid = valid._replace(y=enc.transform(valid.y.reshape(-1, 1)).toarray())
    test = test._replace(y=enc.transform(test.y.reshape(-1, 1)).toarray())

    return train, valid, test
