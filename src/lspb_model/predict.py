import logging
import typing as t
import pandas as pd
from lspb_model.config.core import config
from lspb_model.processing import data_management
from lspb_model import __version__ as _version
from lspb_model.processing.validation import validate_inputs

_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_price_pipe = data_management.load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, test_data: t.Union[pd.DataFrame, t.Dict[str, str]]):
    """Make a prediction using a saved model pipeline."""
    # more checks can be here if for example we deal with completely raw data

    data = pd.DataFrame(test_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _price_pipe.predict(X=validated_data[config.model_config.features])
        _logger.info(f"Making predictions with model version: {_version} "
                     f"Predictions: {predictions}")

        results = {"predictions": predictions, "version": _version, "errors": errors}

    return results
