import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


class Encoder(layers.Layer):
    def __init__(self, latent_dim, original_dim, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.flatten = keras.layers.Flatten()
        self.dense = layers.Dense(units=latent_dim, activity_regularizer=regularizers.l1())

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense(x)
        x = x / (tf.norm(x, axis=-1, keepdims=True) + 1e-9)
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
            latent_dim=6,
            name="autoencoder",
            **kwargs
    ):
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, original_dim=original_dim)
        self.decoder = Decoder(original_dim)

    def call(self, inputs):
        concepts = self.encoder(inputs)
        reconstructed = self.decoder(concepts)
        return reconstructed