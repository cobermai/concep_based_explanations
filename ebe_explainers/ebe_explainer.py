""""
Code from this file is inspired by https://github.com/nesl/ExMatchina
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import Model, layers


def separate_model(model: Model):
    """
    separate last layer from model and return two separate models
    """
    num_layers = len(model.layers)
    feature_model = Model(inputs=model.input,
                          outputs=model.layers[num_layers - 2].output)
    pred_model_shape = model.layers[num_layers - 2].output.shape
    pred_model_shape = pred_model_shape[1:]  # Remove Batch from front.
    pred_model_input = layers.Input(shape=pred_model_shape)
    x = pred_model_input
    for layer in model.layers[num_layers - 1:]:  # idx + 1
        x = layer(x)
    pred_model = Model(pred_model_input, x)
    return feature_model, pred_model


def get_examples(model: Model, train, test, n_concepts, metric="euc_dist"):

    feature_model, pred_model = separate_model(model)
    embeddings = feature_model.predict(train.X)

    kmeans = KMeans(n_clusters=n_concepts, random_state=0).fit(embeddings)
    centers = kmeans.cluster_centers_
    k = 1 # find closest k signals to center
    X_examples = np.zeros((n_concepts*k, len(train.X[0]), len(train.X[0, 0])))
    y_examples = np.zeros((n_concepts*k))
    for i, center in enumerate(centers):
        if metric == "cosine":
            distance = cosine_similarity(embeddings, centers[i:i + 1, :])[:, 0]
            indices = np.argsort(distance)[-k:]
        elif metric == "dot":
            distance = (embeddings @ centers[i:i + 1, :].T)[:, 0]
            indices = np.argsort(distance)[:k]
        elif metric == "euc_dist":
            distance = np.linalg.norm(embeddings - centers[i:i + 1, :], axis=1)
            indices = np.argsort(distance)[:k]

        X_examples[i:i+k] = test.X[indices]

    y_examples = model.predict(X_examples)

    return X_examples, y_examples