import matplotlib.pyplot as plt
import numpy as np


def plot_concepts_by_label(concepts, output_dir, y_onehot, n_labels=2):
    """
    function plots 1d concepts with labels as title
    """
    y = np.argmax(y_onehot, axis=1)
    fig, ax = plt.subplots(1, n_labels, figsize=(4 * n_labels, 3))

    for n_label, concept in enumerate(concepts):
        ax[y[n_label]].plot(concept, label=f"concept {n_label}")
        ax[y[n_label]].set_title(f"class {y[n_label]}")
        ax[y[n_label]].legend()
        ax[y[n_label]].set_xlabel('x')
        ax[y[n_label]].set_ylabel('y')
        ax[y[n_label]].set_ylim((np.min(concepts) * 1.1, np.max(concepts) * 1.1))
    plt.tight_layout()
    plt.savefig(output_dir)

def plot_concepts2d(concepts, output_dir, show=False, labels=[]):
    """
    function plots 2d concepts with labels as title
    """
    fig, ax = plt.subplots(1,len(concepts), figsize=(20,4))
    for i, ax in enumerate(ax.flatten()):
        plottable_image = concepts[i]
        plt.gray()
        ax.imshow(plottable_image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if len(labels) > 0:
            ax.set_title(f"concept {i} \n label: {labels[i]}")
        else:
            ax.set_title(f"concept {i}")
    if show:
        plt.show()

    plt.savefig(output_dir)
