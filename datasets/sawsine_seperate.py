from pathlib import Path
import numpy as np
from dataset import DatasetCreator
import matplotlib.pyplot as plt


class SAWSINE_SEPERATE(DatasetCreator):
    """
    Subclass of DatasetCreator to specify dataset selection. None of the abstract functions from abstract class can
    be overwritten.
    """

    @staticmethod
    def get_data(filepath) -> tuple:
        """
        return artificial data
        """
        len_ds = 8000  # default 8000
        noise_level = 1.1#1.1  # default 1.1
        X = np.ones((len_ds, 110, 1))
        y = np.hstack((np.zeros(int(len_ds / 2)), np.ones(int(len_ds / 2))))

        def add_noise(X, noise_level):
            return X * (1 - np.random.rand(110) * noise_level) + np.random.rand(110) * noise_level

        for i in range(int(len_ds / 4)):
            X[i, :, 0] = add_noise(np.sin(np.arange(0, 11, 0.1)), noise_level)
            X[i + int(len_ds / 4), :, 0] = add_noise(np.abs(np.sin(np.arange(0, 11, 0.1))), noise_level)

            X[i + int(len_ds / 2), :, 0] = add_noise(
                np.hstack((np.zeros(30), np.ones(50) * np.arange(50) / -50, np.zeros(30))), noise_level)
            X[i + int(len_ds / 4 * 3), :, 0] = add_noise(np.hstack((np.zeros(20), np.ones(30), np.zeros(60))),
                                                         noise_level)
        return X, y

if __name__ == '__main__':
    X, y = SAWSINE_SEPERATE.get_data(filepath=Path(""))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(X[y == 0][:, :, 0].T, alpha=0.1)
    ax[0].plot(X[y == 0][0, :, 0].T)
    ax[0].plot(X[y == 0][0, :, 0].T)
    ax[0].set_title("class 0")
    ax[1].plot(X[y == 1][:, :, 0].T, alpha=0.1)
    ax[1].plot(X[y == 1][0, :, 0].T)
    ax[1].plot(X[y == 1][0, :, 0].T)
    ax[1].set_title("class 1")
    plt.show()