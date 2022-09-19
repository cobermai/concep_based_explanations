"""
model setup according to https://www.tensorflow.org/guide/keras/custom_layers_and_models
"""
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow import keras
from tensorflow.keras import Input

from classifiers import fcn


class Classifier:
    """
    Classifier class which acts as wrapper for tensorflow models.
    """

    def __init__(self,
                 input_shape: tuple,
                 output_directory: Path,
                 classifier_name: str,
                 num_classes: int,
                 monitor: str,
                 loss: str,
                 optimizer: str,
                 epochs: int,
                 batch_size: int,
                 learning_rate: float,
                 reduce_lr_factor: float,
                 reduce_lr_patience: int,
                 min_lr: float,
                 build=True,
                 output_model_structure=False):
        """
        Initializes the Classifier with specified settings
        :param output_directory: Directory for model output.
        :param classifier_name: Name of classifier, e.g. 'fcn'.
        :param num_classes: number of classes in input
        :param monitor: Name of performance variable to monitor.
        :param loss: Name of loss function to use in training.
        :param optimizer: Name of optimizer used in training.
        :param epochs: Number of epochs.
        :param batch_size: Number of input data used in each batch.
        :param build: Bool stating whether the model is to be build.
        """
        self.input_shape = input_shape
        self.output_directory = output_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        self.classifier_name = classifier_name
        self.num_classes = num_classes
        self.monitor = monitor
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_patience = reduce_lr_patience
        self.min_lr = min_lr
        self.output_model_structure = output_model_structure
        if build:
            self.model = self.build_classifier()
            self.model.build(input_shape)
            if output_model_structure is True:
                keras.utils.plot_model(self.model,
                                       to_file=output_directory /
                                       "plot_model_structure.png",
                                       show_shapes=True,
                                       show_layer_names=True)

    def build_classifier(self, **kwargs):
        """
        Builds classifier model
        **kwargs: Keyword arguments for tf.keras.Model.compile method
        :return: Tensorflow model for classification of time series data
        """
        if self.classifier_name == 'fcn':
            model = fcn.FCNBlock(self.num_classes)
        else:
            raise AssertionError("Model name does not exist")

        metrics = [
            ToCategoricalMetric(keras.metrics.TruePositives, name='tp'),
            ToCategoricalMetric(keras.metrics.FalsePositives, name='fp'),
            ToCategoricalMetric(keras.metrics.TrueNegatives, name='tn'),
            ToCategoricalMetric(keras.metrics.FalseNegatives, name='fn'),
            ToCategoricalMetric(keras.metrics.BinaryAccuracy, name='accuracy'),
            ToCategoricalMetric(keras.metrics.Precision, name='precision'),
            ToCategoricalMetric(keras.metrics.Recall, name='recall'),
            ToCategoricalMetric(keras.metrics.AUC, name='auc', prob_dim=1),
        ]

        # converting the tf subclass model into a functional model. This
        # enables to use the Explainer
        x = Input(shape=self.input_shape[1:])
        model = keras.models.Model(inputs=[x], outputs=model.call(x))

        optimizer = keras.optimizers.get(self.optimizer)
        optimizer.learning_rate = self.learning_rate
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=metrics,
                      **kwargs)

        return model

    @staticmethod
    def get_class_weight(y: np.ndarray) -> dict:
        """
        Function returns class weight for class imbalanced datasets.
        The sum of the weights of all examples stays the same.
        :param y: one hot encoded labels of train set
        return: dict with class weight for each label
        """
        label_indices = np.argmax(y, axis=1)
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(label_indices),
            y=label_indices)
        class_weights_dict = dict(enumerate(class_weights))
        return class_weights_dict

    def fit_classifier(self, train, valid, **kwargs):
        """
        Trains classifier model on input data
        :param train: named tuple containing training set
        :param valid: named tuple containing validation set
        **kwargs: Keyword arguments for tf.keras.Model.fit method
        """
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor=self.monitor,
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            min_lr=self.min_lr)

        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(self.output_directory / 'best_model.hdf5'),
            save_weights_only=True,
            monitor=self.monitor,
            save_best_only=True)

        self.model.fit(x=train.X,
                       y=train.y,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=1,
                       validation_data=(valid.X, valid.y),
                       callbacks=[reduce_lr, model_checkpoint],
                       class_weight=self.get_class_weight(train.y),
                       **kwargs)


class ToCategoricalMetric(tf.keras.metrics.Metric):
    """
    Wrapper function to enable one hot encoded inputs to tf.metrics
    """

    def __init__(self, binary_metric, prob_dim=False, **kwargs):
        """
        :param binary_metric: tf.metrics which takes binary inputs
        """
        super(ToCategoricalMetric, self).__init__(**kwargs)
        self.binary_metric = binary_metric(**kwargs)
        self.prob_dim = prob_dim

    def update_state(self, y_true, y_pred, **kwargs):
        """
        updates metric state
        :param y_true: one hot encoded true labels
        :param y_true: one hot encoded predicted labels
        """
        y_true = tf.math.argmax(y_true, axis=-1)
        y_true = tf.cast(y_true, tf.bool)

        if self.prob_dim:
            y_pred = y_pred[:, self.prob_dim]
        else:
            y_pred = tf.math.argmax(y_pred, axis=-1)
            y_pred = tf.cast(y_pred, tf.bool)

        self.binary_metric.update_state(y_true, y_pred, **kwargs)

    def reset_states(self):
        """ resets metric state """
        return self.binary_metric.reset_states()

    def result(self):
        """ returns current metric state """
        return self.binary_metric.result()
