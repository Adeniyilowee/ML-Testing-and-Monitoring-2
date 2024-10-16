import pytest
from lspb_model.config.core import config
from lspb_model.processing.data_management import load_dataset, load_testdataset


@pytest.fixture(scope="session")
def pipeline_inputs():
    # For larger datasets, here we would use a testing sub-sample.
    data = load_dataset(file_name=config.app_config.training_data_file)
    return data


@pytest.fixture()
def raw_training_data():
    # For larger datasets, here we would use a testing sub-sample.
    return load_dataset(file_name=config.app_config.training_data_file)


@pytest.fixture()
def sample_input_data():
    return load_testdataset(file_name=config.app_config.test_data_file)
