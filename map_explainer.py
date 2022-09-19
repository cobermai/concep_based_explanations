import math
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans

from map_explainers import ae1d_1e_3d
from map_explainers import ae2d_4ec_4dc


class Explainer:
    """
    Model agnostic explainer class.
    """

    def __init__(self,
                 input_shape: tuple,
                 output_directory: Path,
                 epochs: int,
                 batch_size: int,
                 explainer_name: str,
                 n_concepts: int,
                 latent_dim: int,
                 build=True,
                 sep_loss=1
                 ):
        """
        Initializes the explainer with specified settings
        """
        self.input_shape = input_shape
        self.output_directory = output_directory
        self.epochs = epochs
        self.batch_size = batch_size
        self.explainer_name = explainer_name
        self.n_concepts = n_concepts
        self.latent_dim = latent_dim
        self.sep_loss = sep_loss
        if build:
            self.explainer = self.build_explainer()

    def build_explainer(self, **kwargs):
        """
        Builds explainer model
        **kwargs: Keyword arguments for tf.keras.Model.compile method
        :return: Tensorflow model for explanation of time series data
        """
        if self.explainer_name == 'ae1d_1e_3d':
            explainer = ae1d_1e_3d.AutoEncoder(
                original_dim=self.input_shape,
                latent_dim=self.latent_dim)
        elif self.explainer_name == 'ae2d_4ec_4dc':
            explainer = ae2d_4ec_4dc.AutoEncoder(
                original_dim=self.input_shape, latent_dim=self.latent_dim)
        else:
            raise AssertionError("Model name does not exist")

        return explainer

    def fit_explainer(self, classifier, X, explainer_name):
        """
        function fits model-agnostic explainer
        :param classifier: classifier to explain
        :param X: data
        :param explainer_name: name of explainer
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        ce_loss = tf.keras.losses.CategoricalCrossentropy()
        mse_loss = tf.keras.losses.MeanSquaredError()
        loss_metric = tf.keras.metrics.Mean()

        classifier.trainable = False
        y_pred_train = classifier.predict(X)

        bat_per_epoch = math.floor(len(X) / self.batch_size)
        # Iterate over epochs.
        for epoch in range(self.epochs):
            print("Start of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step in range(bat_per_epoch):

                n = step * self.batch_size
                # generate training batch
                x_batch_train = X[n:n + self.batch_size].astype(np.float32)
                y_batch_train = y_pred_train[n:n + self.batch_size]

                with tf.GradientTape() as tape:
                    enc_out = self.explainer.encoder(x_batch_train)

                    latent = enc_out

                    x_reconstructed = self.explainer.decoder(latent)
                    y_pred_reconstructed = classifier(x_reconstructed)

                    # Loss
                    sep_reg_new = self.cluster_regularization(
                        x_batch_train) * self.sep_loss
                    mse_loss_new = mse_loss(x_batch_train, x_reconstructed)
                    ce_loss_new = ce_loss(y_batch_train, y_pred_reconstructed)
                    loss = mse_loss_new + ce_loss_new + sep_reg_new

                grads = tape.gradient(loss, self.explainer.trainable_weights)
                optimizer.apply_gradients(
                    zip(grads, self.explainer.trainable_weights))

                loss_metric(loss)
                if step % 100 == 0:
                    print(f"loss:{loss.numpy()}, sep_reg:{sep_reg_new.numpy()}"
                          f", mse_loss:{mse_loss_new.numpy()}"
                          f", ce_loss:{ce_loss_new.numpy()}")

                pd.DataFrame({"epoch": epoch}, index=[0]).to_csv(
                    self.output_directory / "epoch.csv")

        self.explainer.save_weights(
            str(self.output_directory / "ae_weights.h5"))

    def cluster_regularization(self, X):
        """
        MAP separation regularization
        :param X: batch data
        :return: separation loss
        """
        all_latent = self.explainer.encoder(X)

        kmeans = KMeans(
            n_clusters=self.n_concepts,
            random_state=0).fit(
            all_latent.numpy())
        concepts = kmeans.cluster_centers_

        sep_loss = self.separation_regularization(concepts)
        return sep_loss

    def separation_regularization(self, concepts):
        """
        function from https://github.com/chihkuanyeh/concept_exp
        calculate Second regularization term, i.e. the similarity between concepts, to be minimized
        Note: it is important to MAKE SURE L2 GOES DOWN! that will let concepts separate from each other
        :param concepts: extracted concepts
        """
        all_concept_dot = tf.transpose(concepts) @ concepts
        mask = np.eye(len(concepts[0])) * -1 + 1  # the i==j positions are 0
        L_sparse_2_new = tf.reduce_sum(all_concept_dot * mask) \
            / (self.n_concepts * (self.n_concepts - 1)) \
            / tf.size(concepts[:, 0], out_type=tf.float32)

        return L_sparse_2_new

    def get_concepts_kmeans(self, X):
        """
        :param X: data
        :return: reconstructed concepts and their lower dimensional prototypes
        """
        latent = self.explainer.encoder(X)

        kmeans = KMeans(n_clusters=self.n_concepts, random_state=0).fit(latent)
        centers = kmeans.cluster_centers_

        reconstructed = np.zeros(tuple((self.n_concepts,)) + np.shape(X[0]))
        for i, center in enumerate(centers):
            print(f"values of kmeans center {i}: {center}")
            reconstructed[i] = self.explainer.decoder(np.array([center]))
        return reconstructed, centers
