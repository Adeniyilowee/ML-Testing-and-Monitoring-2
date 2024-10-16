from pathlib import Path

from lspb_model.config.core import create_and_validate_config, fetch_config_from_yaml

import pytest
from pydantic import ValidationError


TEST_CONFIG_TEXT = """
package_name: fullstack_boosting_model
pipeline_name: boosting_pipe
pipeline_save_file: boosting
training_data_file: LandslideData.csv
test_data_file: test1.csv
modelchoice: boosting_dt
variables_to_rename:
  foo: bar
variables_to_reorder:
 - ID
 - ASPECT
 - STREAM_DIST
 - BASE_AREA
 - BASIN
 - CURVATURE
 - CURVE_CONT
 - CURVE_PROF
 - CURVES
 - DROP
 - ROCK_DIST
 - FLOW_DIR
 - FOS
 - LITHOLOGY
 - ELEVATION
 - COHESION
 - SCARP_DIST
 - SCARPS
 - FRICTION_ANGLE
 - SLOPE
 - SLOPE_LEG
 - WOODS
 - SPECIFIC_WT
 - LANDSLIDE
numerical_vars_1:
  - ASPECT
  - BASE_AREA
  - BASIN
  - CURVATURE
  - CURVE_CONT
  - CURVE_PROF
  - CURVES
  - DROP
  - FLOW_DIR
numerical_vars_2:
  - FOS
  - ELEVATION
  - COHESION
  - FRICTION_ANGLE
  - SLOPE
  - SLOPE_LEG
  - SPECIFIC_WT
categorical_vars_1:
  - LANDSLIDE
categorical_vars_2:
  - LITHOLOGY
  - SCARPS
  - WOODS
astype_features:
  LANDSLIDE: 'int64'
  LITHOLOGY': 'int64'
  WOODS': 'int64'
  SCARPS': 'int64'
special_edit: SCARPS
negative_variables:
  - CURVATURE
  - CURVE_CONT
  - CURVE_PROF
variables_to_drop:
  - ASPECT
  - LITHOLOGY
target: LANDSLIDE
features:
  - STREAM_DIST
  - BASE_AREA
  - BASIN
  - CURVATURE
  - CURVE_CONT
  - CURVE_PROF
  - CURVES
  - DROP
  - ROCK_DIST
  - FLOW_DIR
  - FOS
  - ELEVATION
  - COHESION
  - SCARP_DIST
  - SCARPS
  - FRICTION_ANGLE
  - SLOPE
  - SLOPE_LEG
  - WOODS
  - SPECIFIC_WT
test_size: 0.1
train_size: 0.9
random_state: 0
n_estimators: 50
loss: ls
allowed_loss_functions:
  - ls
  - huber
"""

INVALID_TEST_CONFIG_TEXT = """
package_name: fullstack_boosting_model
pipeline_name: boosting_pipe
pipeline_save_file: boosting
training_data_file: LandslideData.csv
test_data_file: test1.csv
variables_to_reorder:
 - ID
 - ASPECT
 - STREAM_DIST
 - BASE_AREA
 - BASIN
 - CURVATURE
 - CURVE_CONT
 - CURVE_PROF
 - CURVES
 - DROP
 - ROCK_DIST
 - FLOW_DIR
 - FOS
 - LITHOLOGY
 - ELEVATION
 - COHESION
 - SCARP_DIST
 - SCARPS
 - FRICTION_ANGLE
 - SLOPE
 - SLOPE_LEG
 - WOODS
 - SPECIFIC_WT
 - LANDSLIDE
numerical_vars_1:
  - ASPECT
  - BASE_AREA
  - BASIN
  - CURVATURE
  - CURVE_CONT
  - CURVE_PROF
  - CURVES
  - DROP
  - FLOW_DIR
numerical_vars_2:
  - FOS
  - ELEVATION
  - COHESION
  - FRICTION_ANGLE
  - SLOPE
  - SLOPE_LEG
  - SPECIFIC_WT
categorical_vars_1:
  - LANDSLIDE
categorical_vars_2:
  - LITHOLOGY
  - SCARPS
  - WOODS
astype_features:
  LANDSLIDE: 'int64'
  LITHOLOGY': 'int64'
  WOODS': 'int64'
  SCARPS': 'int64'
special_edit: SCARPS
negative_variables:
  - CURVATURE
  - CURVE_CONT
  - CURVE_PROF
variables_to_drop:
  - ASPECT
  - LITHOLOGY
target: LANDSLIDE
features:
  - STREAM_DIST
  - BASE_AREA
  - BASIN
  - CURVATURE
  - CURVE_CONT
  - CURVE_PROF
  - CURVES
  - DROP
  - ROCK_DIST
  - FLOW_DIR
  - FOS
  - ELEVATION
  - COHESION
  - SCARP_DIST
  - SCARPS
  - FRICTION_ANGLE
  - SLOPE
  - SLOPE_LEG
  - WOODS
  - SPECIFIC_WT
test_size: 0.1
train_size: 0.9
random_state: 0
n_estimators: 50
loss: ls
allowed_loss_functions:
  - huber
"""


def test_fetch_config_structure(tmpdir):

    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    config_1 = configs_dir / "sample_config.yml"  # create an empty yaml file
    config_1.write_text(TEST_CONFIG_TEXT)  # then write our string to the file
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)  # we simply just fetch it afterwards

    config = create_and_validate_config(parsed_config=parsed_config)

    assert config.model_config
    assert config.app_config


def test_config_validation_raises_error_for_invalid_config(tmpdir):

    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    config_1 = configs_dir / "sample_config.yml"

    config_1.write_text(INVALID_TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)

    with pytest.raises(ValidationError) as excinfo:
        create_and_validate_config(parsed_config=parsed_config)

    assert "not in the allowed set" in str(excinfo.value)


def test_missing_config_field_raises_validation_error(tmpdir):

    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    config_1 = configs_dir / "sample_config.yml"
    TEST_CONFIG_TEXT = """package_name: fullstack_boosting_model"""
    config_1.write_text(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)

    with pytest.raises(ValidationError) as excinfo:
        create_and_validate_config(parsed_config=parsed_config)

    assert "field required" in str(excinfo.value)
    assert "pipeline_name" in str(excinfo.value)
