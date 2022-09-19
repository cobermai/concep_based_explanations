from tensorflow.keras import layers


class CNNBlock(layers.Layer):
    """Convolutional Neural Net Block"""
    def __init__(self, filters, kernel_size):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.Activation(activation='relu')
        self.filters = filters
        self.kernel_size = kernel_size

    def call(self, input_tensor, training=None, mask=None):
        x = self.conv(input_tensor)
        x = self.bn(x, training=False)
        x = self.relu(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config
