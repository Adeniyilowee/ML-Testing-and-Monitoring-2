from sklearn.metrics import accuracy_score
from lspb_model import predict


def test_predict(sample_input_data):
    test_inputs = sample_input_data
    prediction = predict.make_prediction(test_data=test_inputs.iloc[:, :-1])
    accuracy = accuracy_score(test_inputs['LANDSLIDE'], prediction.get("predictions"))
    assert accuracy > 0.90
