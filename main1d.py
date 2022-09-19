import json
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from classifier import Classifier
from dataset import load_dataset
from datasets.ecg200 import ECG200
from datasets.sawsine_seperate import SAWSINE_SEPERATE
from ebe_explainers.ebe_explainer import get_examples, separate_model
from map_explainer import Explainer
from msp_explainer import PrototypeExplainer
from utils.utils import ConceptProperties
from visualization import viszualization

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def train_classifier(
        train: namedtuple,
        valid: namedtuple,
        test: namedtuple,
        output_dir,
        filepath) -> tf.keras.Model:
    """
    Function trains black box model
    :param train: training data
    :param test: test data
    :param output_dir: path to output results
    :param n_concepts: number of concepts to extract
    :return: classifier
    """
    # Load hyperparameters classifier
    classifier_hyperparameters = open(
        filepath / "classifiers/default_hyperparameters.json")
    classifier_hp_dict = json.load(classifier_hyperparameters)

    # define and fit classifier
    clf = Classifier(
        input_shape=np.shape(
            train.X),
        output_directory=output_dir,
        **classifier_hp_dict)
    fit_classifier = not Path(output_dir / 'best_model.hdf5').exists()
    if fit_classifier:
        clf.fit_classifier(train, valid)
    clf.model.load_weights(str(output_dir / 'best_model.hdf5'))

    # eval classifier
    results = clf.model.evaluate(x=test.X, y=test.y, return_dict=True)
    pd.DataFrame.from_dict(
        results, orient='index').T.to_csv(
        output_dir / "results_classification.csv")

    return clf.model


def get_ebe_explanations(
        model: tf.keras.Model,
        train: namedtuple,
        test: namedtuple,
        output_dir: Path,
        n_concepts: int):
    """
    Function extracts explanation by example from trained model and stores all concepts and their properties into
    output folder
    :param train: training data
    :param test: test data
    :param output_dir: path to output results
    :param n_concepts: number of concepts to extract
    """
    #
    output_dir_exbyex = output_dir / 'ebe'
    output_dir_exbyex.mkdir(parents=True, exist_ok=True)
    X_examples, y_examples = get_examples(
        model=model, train=train, test=test, n_concepts=n_concepts)
    viszualization.plot_concepts_by_label(
        X_examples,
        output_dir /
        "example_explanations.png",
        y_examples)

    # get concept properties
    cp = ConceptProperties()

    feature_model, pred_model = separate_model(model)
    latent = feature_model(test.X)
    concepts = feature_model(X_examples)

    y_pred = model(test.X)
    instance_concept = cp.get_closest_concept_to_instances(
        latent.numpy(), concepts.numpy())
    y_pred_reconstructed = pred_model(instance_concept)

    completness = cp.get_completness(y_pred, y_pred_reconstructed)
    conceptSHAP = cp.get_concept_shap(predictor=pred_model,
                                      decoder=None,
                                      latent=latent.numpy(),
                                      concepts=concepts.numpy(),
                                      y_pred=y_pred.numpy())
    pd.DataFrame({
        "model": "EBE",
        "accuracy": cp.get_completness(test.y, y_pred),
        "output_dir": output_dir,
        "n_concepts": n_concepts,
        "completness": completness,
        "conceptSHAP": [conceptSHAP],
    }, index=[0]) \
        .to_csv(output_dir / "completeness_importance_ebe.csv")


def get_msp_explanations(
        train: namedtuple,
        test: namedtuple,
        output_dir: Path,
        n_concepts: int,
        epochs: int):
    """
    Function trains model-specific explainer and stores all concepts and their properties into output folder
    :param train: training data
    :param test: test data
    :param output_dir: path to output results
    :param n_concepts: number of concepts to extract
    :param epochs: number of epochs to train map
    """
    output_dir_protoex = output_dir / 'msp'
    output_dir_protoex.mkdir(parents=True, exist_ok=True)

    pexp = PrototypeExplainer(input_shape=np.shape(train.X[0]),
                              output_directory=output_dir_protoex,
                              n_concepts=n_concepts,
                              latent_dim=n_concepts,
                              n_classes=2,
                              explainer_name='prototype1',
                              epochs=epochs,
                              batch_size=32)
    fit_explainer = not Path(output_dir_protoex / "msp.h5").exists()
    if fit_explainer:
        pexp.fit_explainer(X=train.X, y=train.y)
    else:
        pexp.explainer.build(input_shape=np.shape(train.X))
        pexp.explainer.load_weights(str(output_dir_protoex / 'msp.h5'))

    reconstructed_prototypes = pexp.get_reconstructed_prototypes()
    prototypes = pexp.explainer.predictor.prototypes
    prototypes_labels = pexp.explainer.predictor(prototypes)
    viszualization.plot_concepts_by_label(
        reconstructed_prototypes,
        output_dir / "msp.png",
        prototypes_labels)

    # completeness & importance
    cp = ConceptProperties()
    latent = pexp.explainer.encoder(test.X)
    y_pred = pexp.explainer.predictor(latent)
    instance_concept = cp.get_closest_concept_to_instances(
        latent.numpy(), prototypes.numpy())
    y_pred_reconstructed = pexp.explainer.predictor(instance_concept)

    completness = cp.get_completness(y_pred, y_pred_reconstructed)
    conceptSHAP = cp.get_concept_shap(predictor=pexp.explainer.predictor,
                                      decoder=None,  # latent space needs no decoding before prediction
                                      latent=latent.numpy(),
                                      concepts=prototypes.numpy(),
                                      y_pred=y_pred)
    pd.DataFrame({
        "model": "MSP",
        "accuracy": cp.get_completness(test.y, y_pred),
        "n_concepts": n_concepts,
        "output_dir": output_dir,
        "completness": completness,
        "conceptSHAP": [conceptSHAP],
        "latent_centers": [prototypes.numpy()]
    }, index=[0]) \
        .to_csv(output_dir / "completeness_importance_msp.csv")


def get_map_explanations(
        model: tf.keras.Model,
        train: namedtuple,
        test: namedtuple,
        output_dir: Path,
        explainer_name: str,
        n_concepts: int,
        epochs: int):
    """
    Function trains model-agnostic explainer and stores all concepts and their properties into output folder
    :param model: classifier to explain
    :param train: training data
    :param test: test data
    :param output_dir: path to output results
    :param explainer_name: name of map to load
    :param n_concepts: number of concepts to extract
    :param epochs: number of epochs to train map
    """
    output_dir_ex = output_dir / explainer_name
    output_dir_ex.mkdir(parents=True, exist_ok=True)

    exp = Explainer(input_shape=np.shape(train.X[0]),
                    output_directory=output_dir_ex,
                    n_concepts=n_concepts,
                    latent_dim=n_concepts * 5,
                    explainer_name=explainer_name,
                    epochs=epochs,
                    batch_size=32)

    fit_explainer = not Path(output_dir_ex / "map.h5").exists()
    if fit_explainer:
        exp.fit_explainer(
            classifier=model,
            X=train.X,
            explainer_name=explainer_name)
    else:
        exp.explainer.build(input_shape=np.shape(train.X))
        exp.explainer.load_weights(str(output_dir_ex / 'map.h5'))

    X_concepts_kmeans, latent_centers = exp.get_concepts_kmeans(train.X)
    concept_labels = model(X_concepts_kmeans)
    viszualization.plot_concepts_by_label(
        X_concepts_kmeans,
        output_dir_ex / "map.png",
        concept_labels)

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
        concepts=latent_centers,
        y_pred=y_pred)

    pd.DataFrame({
        "model": "MAP",
        "accuracy": cp.get_completness(test.y, y_pred),
        "output_dir": output_dir,
        "n_concepts": n_concepts,
        "completness": completness,
        "conceptSHAP": [conceptSHAP],
        "latent_centers": [latent_centers]
    }, index=[0]) \
        .to_csv(output_dir / "completeness_importance_concept_map.csv")


if __name__ == '__main__':
    # Set file paths
    filepath = Path("")
    # Set parameter grid
    datasets = [ECG200, SAWSINE_SEPERATE]
    explainer_names = ["ae1d_1e_3d"]
    param_grid = {
        "datasets": datasets,
        "explainer_names": explainer_names,
        "n_concepts": [2, 3, 4, 5, 6, 7, 8]
    }
    vary_values = list(map(param_grid.get, param_grid.keys()))
    meshgrid = np.array(np.meshgrid(*vary_values)
                        ).T.reshape(-1, len(param_grid.keys()))
    df_meshgrid = pd.DataFrame(meshgrid, columns=param_grid.keys())

    for index, row in df_meshgrid.iterrows():

        output_dir = filepath / Path("output") / (str(row["n_concepts"]) + '_'
                                                  + row["datasets"].__name__
                                                  + datetime.now().strftime("%Y-%m-%dT%H.%M.%S.%f"))
        output_dir.mkdir(parents=True, exist_ok=True)

        train, valid, test = load_dataset(
            creator=row["datasets"], filepath=filepath / 'datasets')

        n_concepts = row["n_concepts"]
        epochs = 1500

        model = train_classifier(
            train=train,
            valid=valid,
            test=test,
            output_dir=output_dir,
            filepath=filepath)

        get_ebe_explanations(model=model,
                             train=train,
                             test=test,
                             output_dir=output_dir,
                             n_concepts=n_concepts)

        get_map_explanations(
            model=model,
            train=train,
            test=test,
            output_dir=output_dir,
            explainer_name=row["explainer_names"],
            n_concepts=n_concepts,
            epochs=epochs)

        tf.keras.backend.clear_session()

        get_msp_explanations(
            train=train,
            test=test,
            output_dir=output_dir,
            n_concepts=n_concepts,
            epochs=epochs)
        tf.keras.backend.clear_session()
