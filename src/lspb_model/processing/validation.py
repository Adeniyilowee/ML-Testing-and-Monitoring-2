from lspb_model.config.core import config
import pandas as pd
from marshmallow import fields, Schema, ValidationError


class LandslideSchema(Schema):
    STREAM_DIST = fields.Float()
    BASE_AREA = fields.Float()
    BASIN = fields.Float()
    CURVATURE = fields.Float()
    CURVE_CONT = fields.Float()
    CURVE_PROF = fields.Float()
    CURVES = fields.Float()
    DROP = fields.Float()
    ROCK_DIST = fields.Float()
    FLOW_DIR = fields.Float()
    FOS = fields.Float()
    ELEVATION = fields.Float()
    COHESION = fields.Float()
    SCARP_DIST = fields.Float()
    SCARPS = fields.Integer()
    FRICTION_ANGLE = fields.Float()
    SLOPE = fields.Float()
    SLOPE_LEG = fields.Float()
    WOODS = fields.Integer()
    SPECIFIC_WT = fields.Float()


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    if input_data[config.model_config.features].isnull().any().any():
        validated_data = validated_data.dropna(axis=0, subset=config.model_config.features)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame):
    """Check model inputs for unprocessable values."""

    # convert syntax error field names (beginning with numbers)
    validated_data = drop_na_inputs(input_data=input_data)

    # set many=True to allow passing in a list
    schema = LandslideSchema(many=True)
    errors = None

    try:
        # replace numpy nans so that Marshmallow can validate
        schema.load(validated_data.to_dict(orient="records"))
    except ValidationError as exc:
        errors = exc.messages

    return validated_data, errors
