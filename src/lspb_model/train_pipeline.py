import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

from lspb_model import pipeline
from lspb_model import __version__ as _version
from lspb_model.processing.data_management import load_dataset, load_testdataset, save_pipeline
from lspb_model.config.core import config

import logging

_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    data = load_dataset(file_name=config.app_config.training_data_file)

    piper = pipeline.landscape_pipe.fit_transform(data)

    _logger.warning(f"saving model version:{_version}")
    save_pipeline(pipeline_to_persist=piper)

    test_inputs = load_testdataset(file_name=config.app_config.test_data_file)

    predictions = piper.predict(test_inputs[config.model_config.features])
    predictions_proba = piper.predict_proba(test_inputs[config.model_config.features])[::, 1]
    result_plot(predictions, predictions_proba, test_inputs['LANDSLIDE'])


def result_plot(prediction, prediction_proba, target):
    accuracy = accuracy_score(target, prediction)
    # classification_report(target, prediction)
    cm = confusion_matrix(target, prediction)

    fpr, tpr, _ = roc_curve(target, prediction_proba)
    auc = roc_auc_score(target, prediction_proba)
    confusion_mat(cm)

    plt.figure()
    plt.plot(fpr, tpr, color='green', lw=2, label="{cl_name}, AUC Test Score = {auc}".format(cl_name=config.app_config.pipeline_save_file, auc=str(auc)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Graph | Accuracy = {accuracy}".format(accuracy=str(accuracy)))
    plt.legend(loc=4)
    plt.savefig('images/roc.png')


def confusion_mat(cm):
    confusion_matrix_ = np.array(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Landslide', 'Landslide'],
                yticklabels=['No Landslide', 'Landslide'])

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('images/confusion_matrix.png')


if __name__ == "__main__":
    run_training()
