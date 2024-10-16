import pandas as pd
from lspb_model.config.core import DATASET_DIR

from sklearn.base import BaseEstimator, TransformerMixin


class ToNumeric(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.t = 1+1

    def fit(self, X, y=None):

        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """

        return self

    def transform(self, X):

        dfr = X.iloc[:, 1:]
        X = dfr.apply(pd.to_numeric, downcast='float', errors='coerce')

        return X


class SklearnTransformerWrapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None, transformer=None):

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        The `fit` method allows scikit-learn transformers to
        learn the required parameters from the training data set.
        """

        self.transformer.fit(X[self.variables])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X[self.variables] = self.transformer.transform(X[self.variables])

        return X


class DropNA(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.t = 1+1

    def fit(self, X, y=None):

        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """

        return self

    def transform(self, X):

        X = X.dropna(axis=0, how='any')

        return X


class Astype_features(BaseEstimator, TransformerMixin):

    def __init__(self, astype_features=None):
        self.variables = astype_features

    def fit(self, X, y=None):
        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """

        return self

    def transform(self, X):

        X = X.astype(self.variables)

        return X


class Scarps_Special_Edit(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """
        return self

    def transform(self, X):
        con0 = X[self.variables] == -1
        con1 = X[self.variables] == 23
        con2 = X[self.variables] == 17
        con3 = X[self.variables] == 25

        index = X[con0 | con1 | con2 | con3].index
        X.drop(axis=0, labels=index, inplace=True)

        return X


class NegativeValueEstimator(BaseEstimator, TransformerMixin):
    """dealing with negative values"""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """

        return self

    def transform(self, X):

        for col in self.variables:
            X[col] = X[col] - X[col].min()

        return X


class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        if not isinstance(variables_to_drop, list):
            self.variables = [variables_to_drop]
        else:
            self.variables = variables_to_drop

    def fit(self, X, y=None):
        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """

        return self

    def transform(self, X):

        X = X.drop(self.variables, axis=1)

        return X


class Train_Test_Split(BaseEstimator, TransformerMixin):
    """dealing with negative values"""

    def __init__(self, features=None, target=None, train_size=None, test_size=None, random_state=None):

        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features

        self.target = target
        self.train_size = train_size
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """

        return self

    def transform(self, X):

        if self.random_state:
            X = X.sample(frac=1, random_state=self.random_state)
        else:
            X = X.sample(frac=1)

        test_set_size = int(len(X) * self.test_size)

        train_df = X[:-test_set_size]
        test_df = X[-test_set_size:]

        assert self.features == train_df.iloc[:, :-1].columns.to_list()
        assert self.target in train_df[['LANDSLIDE']].columns.to_list()

        assert self.features == test_df.iloc[:, :-1].columns.to_list()
        assert self.target in test_df[['LANDSLIDE']].columns.to_list()

        data_dir = DATASET_DIR / 'test1.csv'
        test_df.to_csv(data_dir, index=False)
        return train_df
