import math
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from msp_explainers import prototype1
from msp_explainers.autoencoder_helpers import list_of_distances


class PrototypeExplainer:
    """
    Model specific prototype class, proposed by https://github.com/OscarcarLi/PrototypeDL
    """

    def __init__(self,
                 input_shape: tuple,
                 output_directory: Path,
                 epochs: int,
                 batch_size: int,
                 explainer_name: str,
                 n_concepts: int,
                 n_classes: int,
                 latent_dim: int,
                 build=True
                 ):
        """
        Initializes the model with specified settings
        """
        self.input_shape = input_shape
        self.output_directory = output_directory
        self.epochs = epochs
        self.batch_size = batch_size
        self.explainer_name = explainer_name
        self.n_concepts = n_concepts
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        if build:
            self.explainer = self.build_explainer()

    def build_explainer(self, **kwargs):
        """
        Builds explainer model
        **kwargs: Keyword arguments for tf.keras.Model.compile method
        :return: Tensorflow model for explanation of time series data
        """
        if self.explainer_name == 'prototype1':
            explainer = prototype1.AutoEncoder(original_dim=self.input_shape,
                                               latent_dim=self.latent_dim,
                                               n_concepts=self.n_concepts,
                                               n_classes=self.n_classes)
        else:
            raise AssertionError("Model name does not exist")

        return explainer

    def fit_explainer(self, X, y):
        """
        function fits model-specific explainer
        :param X: data
        :param y: data labels
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        ce_loss = tf.keras.losses.CategoricalCrossentropy()
        mse_loss = tf.keras.losses.MeanSquaredError()
        loss_metric = tf.keras.metrics.Mean()

        bat_per_epoch = math.floor(len(X) / self.batch_size)
        # Iterate over epochs.
        for epoch in range(self.epochs):
            print("Start of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step in range(bat_per_epoch):

                n = step * self.batch_size
                # generate training batch
                x_batch_train = X[n:n + self.batch_size].astype(np.float32)
                y_batch_train = y[n:n + self.batch_size]

                with tf.GradientTape() as tape:
                    latent = self.explainer.encoder(x_batch_train)

                    x_reconstructed = self.explainer.decoder(latent)
                    y_pred_reconstructed = self.explainer.predictor(latent)

                    # Loss
                    gamma = 0.05
                    r1_reg_new = self.R1_regularization(latent) * gamma
                    r2_reg_new = self.R2_regularization(latent) * gamma
                    pdl_new = self.PDL1() * 1000

                    mse_loss_new = mse_loss(
                        x_batch_train, x_reconstructed) * gamma
                    ce_loss_new = ce_loss(y_batch_train, y_pred_reconstructed)
                    loss = mse_loss_new + ce_loss_new + r1_reg_new + r2_reg_new + pdl_new

                grads = tape.gradient(loss, self.explainer.trainable_weights)
                optimizer.apply_gradients(
                    zip(grads, self.explainer.trainable_weights))

                loss_metric(loss)
                loss_dict = {
                    'loss': loss.numpy(),
                    'mse_loss': mse_loss_new.numpy(),
                    'ce_loss_new': ce_loss_new.numpy(),
                    'pdl': pdl_new.numpy(),
                    'r1_reg_new': r1_reg_new.numpy(),
                    'r2_reg_new': r2_reg_new.numpy(),
                    'accuracy': self.get_accuracy(X, y)
                }
                if step % 100 == 0:
                    print(loss_dict)

        pd.DataFrame(loss_dict, index=[0]).to_csv(
            str(self.output_directory / "pexp_loss.csv"))
        print(self.explainer.predictor.prototypes.numpy())
        self.explainer.save_weights(
            str(self.output_directory / "pexp_weights.h5"))

    def R1_regularization(self, feature_vectors):
        """
        :param feature_vectors: latent activations of encoder
        :return: R1_regularization
        """
        p = self.explainer.predictor.prototypes
        feature_vector_distances = list_of_distances(p, feature_vectors)
        feature_vector_distances = tf.identity(
            feature_vector_distances, name='feature_vector_distances')
        return tf.reduce_mean(
            tf.reduce_min(
                feature_vector_distances,
                axis=1),
            name='error_1')

    def R2_regularization(self, feature_vectors):
        """
        :param feature_vectors: latent activations of encoder
        :return: R2_regularization
        """
        p = self.explainer.predictor.prototypes
        prototype_distances = list_of_distances(feature_vectors, p)
        prototype_distances = tf.identity(
            prototype_distances, name='prototype_distances')

        return tf.reduce_mean(
            tf.reduce_min(
                prototype_distances,
                axis=1),
            name='error_2')

    def PDL1(self):
        """
        function returns prototype diversity loss according to https://arxiv.org/pdf/1904.08935.pdf
        :return: PDL loss
        """
        p = self.explainer.predictor.prototypes

        distance = []
        for j in range(tf.shape(p)[0] - 1):
            norm = tf.norm(p[j + 1:] - p[j], axis=-1) ** 2
            min_norm = tf.reduce_min(norm).numpy()
            distance.append(min_norm)
        mean_dist = tf.reduce_mean(distance)
        pdl = 1 / (tf.math.log(mean_dist) + 1e-9)
        return pdl

    def get_accuracy(self, X, y):
        """
        function returns accuracy of learned prototypes
        :return: accuracy of learned prototypes
        """
        latent = self.explainer.encoder(X)
        y_pred = self.explainer.predictor(latent)
        accuracy = np.sum(
            np.argmax(
                y_pred,
                axis=1) == np.argmax(
                y,
                axis=1)) / len(y)
        return accuracy

    def get_reconstructed_prototypes(self):
        """
        function returns learned prototypes
        :return: reconstructed prototypes
        """
        p = self.explainer.predictor.prototypes
        reconstructed_prototypes = self.explainer.decoder(p)
        return reconstructed_prototypes

