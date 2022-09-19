from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from dataset import load_dataset, DatasetCreator
from datasets.mnist3_unbalanced import MNIST3_unbalanced
from map_explainer import Explainer
from utils.utils import ConceptProperties
from visualization import viszualization


def main(
        creator: DatasetCreator,
        explainer_name: str,
        output_folder: str,
        sep_loss=1):
    """
    Function executes classification and concept-based explanation of 2d data
    :param creator: dataset to load
    :param explainer_name: name of map_explainer to load and to create output folder
    :param output_folder: folder to store output files
    :param sep_loss: weight value of speration loss
    """
    n_concepts = 10

    filepath = Path('')
    output_dir = Path(filepath) / Path('output') / output_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_ex = output_dir / Path(explainer_name) / Path(
        str(sep_loss) + "_" + datetime.now().strftime("%Y-%m-%dT%H.%M.%S.%f"))
    output_dir_ex.mkdir(parents=True, exist_ok=True)

    train, valid, test = load_dataset(
        creator=creator, filepath=filepath / "datasets")

    input_shape = np.shape(train.X[0])

    # define and fit classifier
    fit_classifier = not Path(output_dir / 'model.h5').exists()
    n_classes = len(np.unique(np.argmax(train.y, axis=1)))
    if fit_classifier:
        model = keras.models.Sequential([
            keras.layers.Conv2D(
                32,
                kernel_size=3,
                activation="relu",
                input_shape=input_shape),
            # filter size? kernel size?
            keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
            keras.layers.MaxPool2D(),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(n_classes, activation="softmax")
        ])
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"])
        history = model.fit(train.X, train.y, epochs=200)
        model.save(output_dir / "model.h5")
    else:
        model = keras.models.load_model(output_dir / "model.h5")

    # eval classifier
    results = model.evaluate(x=test.X, y=test.y, return_dict=True)
    pd.DataFrame.from_dict(
        results,
        orient='index').T.to_csv(
        output_dir_ex /
        "results_classification.csv")

    # define and fit explainer
    exp = Explainer(input_shape=np.shape(train.X[0]),
                    output_directory=output_dir_ex,
                    n_concepts=n_concepts,
                    latent_dim=n_concepts * 5,
                    explainer_name=explainer_name,
                    epochs=1500,
                    batch_size=64,
                    sep_loss=sep_loss)

    fit_explainer = not Path(output_dir_ex / "ae_weights.h5").exists()
    if fit_explainer:
        exp.fit_explainer(
            classifier=model,
            X=train.X,
            explainer_name=explainer_name)

    X_concepts, latent_concepts = exp.get_concepts_kmeans(train.X)
    labels = np.argmax(model.predict(X_concepts), axis=1)
    viszualization.plot_concepts2d(
        X_concepts,
        output_dir_ex /
        "concepts_kmeans.png",
        labels=labels)

    # completeness & importance
    y_pred = model(test.X)
    y_pred_reconstructed = model(exp.explainer(test.X))

    cp = ConceptProperties()
    completness = cp.get_completness(y_pred, y_pred_reconstructed)
    conceptSHAP = cp.get_concept_shap(
        predictor=model,
        decoder=exp.explainer.decoder,
        latent=exp.explainer.encoder(
            test.X).numpy(),
        concepts=latent_concepts,
        y_pred=y_pred)
    pd.DataFrame({
        "model": "MAP",
        "completness": completness,
        "conceptSHAP": [conceptSHAP],
        "latent_centers": [latent_concepts]
    }, index=[0]) \
        .to_csv(output_dir_ex / "completeness_importance_concept.csv")


if __name__ == '__main__':
    # Set parameter grid
    datasets = [MNIST3_unbalanced]
    explainer_names = ['ae2d_4ec_4dc']
    param_grid = {
        "datasets": datasets,
        "explainer_names": explainer_names,
        "sep_loss": [1]
    }
    vary_values = list(map(param_grid.get, param_grid.keys()))
    meshgrid = np.array(np.meshgrid(*vary_values)
                        ).T.reshape(-1, len(param_grid.keys()))
    df_meshgrid = pd.DataFrame(meshgrid, columns=param_grid.keys())

    for index, row in df_meshgrid.iterrows():
        main(creator=row["datasets"],
             explainer_name=row["explainer_names"],
             output_folder=row["datasets"].__name__,
             sep_loss=row["sep_loss"])
