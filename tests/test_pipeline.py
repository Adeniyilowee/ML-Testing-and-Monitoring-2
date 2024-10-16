from lspb_model import pipeline
from lspb_model.config.core import config
from sklearn.metrics import accuracy_score, roc_auc_score


def test_pipeline_drops_unnecessary_features(pipeline_inputs):

    data = pipeline_inputs

    pipeline.landscape_pipe.fit(data)

    transformed_inputs = pipeline.landscape_pipe[:-2].transform(data)

    mf = [feature for feature in config.model_config.variables_to_drop if feature not in transformed_inputs.columns]
    assert len(mf) == 2


def test_pipeline_train_test_split(pipeline_inputs):

    data = pipeline_inputs

    pipeline.landscape_pipe.fit(data)

    train_df = pipeline.landscape_pipe[:-1].transform(data)
    assert config.model_config.features == train_df.iloc[:, :-1].columns.to_list()
    assert config.model_config.target in train_df.columns


def test_pipeline_predict_takes_test_input(pipeline_inputs, sample_input_data):
    data = pipeline_inputs
    piper = pipeline.landscape_pipe.fit_transform(data)

    test_inputs = sample_input_data

    predictions = piper.predict(test_inputs[config.model_config.features])
    predictions_proba = piper.predict_proba(test_inputs[config.model_config.features])[::, 1]
    auc = roc_auc_score(test_inputs['LANDSLIDE'], predictions_proba)
    accuracy = accuracy_score(test_inputs['LANDSLIDE'], predictions)

    assert accuracy > 0.90
    assert auc > 0.95
