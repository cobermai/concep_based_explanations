import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


class Encoder(layers.Layer):
    def __init__(self, latent_dim, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.flatten = keras.layers.Flatten()
        self.conv1 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(2, 2), activation="relu")
        self.conv2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2), activation="relu")
        self.conv3 = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation="sigmoid")
        self.dense = layers.Dense(units=latent_dim, activity_regularizer=regularizers.l1())

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        concepts = self.dense(x)
        concepts_norm = concepts / (tf.norm(concepts, axis=-1, keepdims=True) + 1e-9)
        return concepts_norm

class Decoder(layers.Layer):
    def __init__(self, original_dim, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.stride = 2
        self.dense = layers.Dense(units=original_dim[0] / self.stride * original_dim[1] / self.stride * 64)
        self.conv1 = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same',
                                            strides=(self.stride, self.stride),
                                            activation="relu")
        self.conv2 = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same',
                                            activation="relu")
        self.conv3 = layers.Conv2DTranspose(filters=original_dim[2], kernel_size=(3, 3), padding='same',
                                            activation="sigmoid")
        self.original_dim = original_dim

    def call(self, inputs):
        x = self.dense(inputs)
        x = tf.reshape(x, (-1, int(self.original_dim[0] / self.stride), int(self.original_dim[1] / self.stride), 64))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class AutoEncoder(keras.Model):
    def __init__(
            self,
            original_dim,
            latent_dim=5,
            name="autoencoder",
            **kwargs
    ):
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(original_dim)

    def call(self, inputs):
        concepts = self.encoder(inputs)
        reconstructed = self.decoder(concepts)
        return reconstructed