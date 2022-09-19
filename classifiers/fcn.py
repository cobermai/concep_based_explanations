from abc import ABC
from tensorflow.keras import layers
from tensorflow.keras import Model
from classifiers.layers.cnn import CNNBlock


class FCNBlock(Model, ABC):
    """Fully Convolutional Neural network, initially proposed by https://github.com/hfawaz/dl-4-tsc"""
    def __init__(self, num_classes):
        """
        Initializes FCNBlock
        :param num_classes: Number of classes in data set
        """
        super(FCNBlock, self).__init__()
        self.cnn1 = CNNBlock(filters=128, kernel_size=8)
        self.cnn2 = CNNBlock(filters=256, kernel_size=5)
        self.cnn3 = CNNBlock(filters=128, kernel_size=3)
        self.gap = layers.GlobalAveragePooling1D()
        self.out = layers.Dense(num_classes, activation='softmax')

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
        x = self.out(x)
        return x
