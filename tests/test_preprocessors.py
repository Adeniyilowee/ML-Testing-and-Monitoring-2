from lspb_model.config.core import config
from lspb_model.processing import preprocessors as pp


def test_drop_unnecessary_features_transformer(pipeline_inputs):

    data = pipeline_inputs

    assert all(item in data.columns.to_list() for item in config.model_config.variables_to_drop)

    transformer = pp.DropUnecessaryFeatures(variables_to_drop=config.model_config.variables_to_drop)

    X_transformed = transformer.transform(data)

    assert all(item not in config.model_config.variables_to_drop for item in X_transformed.columns.to_list())
