import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from classifiers.layers.cnn import CNNBlock
from msp_explainers.autoencoder_helpers import list_of_distances


class Encoder(layers.Layer):
    """Fully Convolutional Neural network, initially proposed by https://github.com/hfawaz/dl-4-tsc"""
    def __init__(self, latent_dim, original_dim, name="encoder", **kwargs):
        """
        Initializes FCNBlock
        :param num_classes: Number of classes in data set
        """
        super(Encoder, self).__init__(name=name, **kwargs)
        self.cnn1 = CNNBlock(filters=128, kernel_size=8)
        self.cnn2 = CNNBlock(filters=256, kernel_size=5)
        self.cnn3 = CNNBlock(filters=128, kernel_size=3)
        self.gap = layers.GlobalAveragePooling1D()
        self.dense = layers.Dense(units=latent_dim, activation='relu')


    def call(self, input_tensor, training=None, mask=None):
        """
        Function builds FCN model out of 3 convolutional layers with batch normalization and the relu
        activation function. In the end there is a global average pooling layer which feeds the output into a
        softmax classification layer.
        :param input_tensor: input to model
        :param training: bool for specifying whether model should be training
        :param mask: mask for specifying whether some values should be skipped
        """
        x = self.cnn1(input_tensor)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.gap(x)
        x = self.dense(x)
        return x

class Predictor(layers.Layer):
    def __init__(self, latent_dim, n_concepts, n_classes, name="protonet", **kwargs):
        super(Predictor, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.n_concepts = n_concepts
        self.out = layers.Dense(n_classes, activation='softmax')
        self.prototypes = tf.Variable(tf.random.uniform((n_concepts, latent_dim)),
                                      dtype=np.float32,
                                      trainable=True,
                                      name='prototypes')


    def call(self, inputs): # (batch_size x latent_dim)
        prototype_distances = list_of_distances(inputs, self.prototypes)
        prototype_distances = tf.identity(prototype_distances, name='prototype_distances')
        x = self.out(prototype_distances)

        return x


class Decoder(layers.Layer):
    def __init__(self, original_dim, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.dense = layers.Dense(units=300)
        self.dense1 = layers.Dense(units=300, activation="sigmoid")
        self.dense2 = layers.Dense(units=np.prod(original_dim))

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.dense1(x)
        reconstructed = self.dense2(x)
        reshaped = tf.reshape(reconstructed, tuple((-1,) + self.original_dim))
        return reshaped

class AutoEncoder(keras.Model):
    def __init__(
            self,
            original_dim,
            latent_dim,
            n_concepts,
            n_classes=2,
            name="autoencoder",
            **kwargs
    ):
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, original_dim=original_dim)
        self.predictor = Predictor(latent_dim=latent_dim, n_concepts=n_concepts, n_classes=n_classes)
        self.decoder = Decoder(original_dim)

    def call(self, inputs):
        concepts = self.encoder(inputs)
        prediction = self.predictor(concepts)
        reconstructed = self.decoder(concepts)
        return reconstructed