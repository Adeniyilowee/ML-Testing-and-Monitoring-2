from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin


class Model_Choice(BaseEstimator, TransformerMixin):
    """dealing with negative values"""

    def __init__(self, loss, random_state, n_estimators, modelchoice):

        if modelchoice == "boosting_dt":

            classifier = DecisionTreeClassifier(ccp_alpha=0.0, criterion='entropy',
                                                min_samples_leaf=1, min_samples_split=4,
                                                min_weight_fraction_leaf=0.0, splitter='best')

            self.model = AdaBoostClassifier(estimator=classifier,
                                            n_estimators=n_estimators,
                                            random_state=random_state)

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        self.model.fit(X.iloc[:, :-1], X['LANDSLIDE'])

        return self.model
