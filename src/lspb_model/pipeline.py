from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from lspb_model.processing import preprocessors as pp
import numpy as np
from lspb_model import model as md
from lspb_model.config.core import config

import logging


_logger = logging.getLogger(__name__)

landscape_pipe = Pipeline(
    [
        (
            "to_numeric",
            pp.ToNumeric(),
        ),
        (
            "numerical_imputer_1",
            pp.SklearnTransformerWrapper(variables=config.model_config.numerical_vars_1,
                                         transformer=SimpleImputer(strategy='median')),
        ),
        (
            "numerical_imputer_2",
            pp.SklearnTransformerWrapper(variables=config.model_config.numerical_vars_2,
                                         transformer=SimpleImputer(strategy='mean')),
        ),
        (
            "categorical_imputer_1",
            pp.SklearnTransformerWrapper(variables=config.model_config.categorical_vars_1,
                                         transformer=SimpleImputer(strategy='constant', fill_value=np.nan)),
        ),
        (
            "categorical_imputer_2",
            pp.SklearnTransformerWrapper(variables=config.model_config.categorical_vars_2,
                                         transformer=SimpleImputer(strategy='most_frequent')),
        ),
        (
            "dropna_in_all_variable",
            pp.DropNA(),
        ),
        (
            "apply_astype_encoder",
            pp.Astype_features(astype_features=config.model_config.astype_features),
        ),
        (
            "special_encoder",
            pp.Scarps_Special_Edit(variables=config.model_config.special_edit),
        ),
        (
            "negative_value_encoder",
            pp.NegativeValueEstimator(variables=config.model_config.negative_variables),
        ),
        (
            "drop_features",
            pp.DropUnecessaryFeatures(variables_to_drop=config.model_config.variables_to_drop),
        ),
        (
            "train_test_split",
            pp.Train_Test_Split(features=config.model_config.features,
                                target=config.model_config.target,
                                train_size=config.model_config.train_size,
                                test_size=config.model_config.test_size,
                                random_state=config.model_config.random_state),
        ),
        (
            "model_choice",
            md.Model_Choice(loss=config.model_config.loss,
                            random_state=config.model_config.random_state,
                            n_estimators=config.model_config.n_estimators,
                            modelchoice=config.model_config.modelchoice),
        ),
    ]
)
