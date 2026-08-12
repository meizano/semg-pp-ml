"""
Classical Machine Learning models (scikit-learn based).
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator


class ClassicalModelWrapper(BaseEstimator):
    """Wrapper to make sklearn models compatible with Keras-like interface."""
    def __init__(self, model: BaseEstimator, input_shape: tuple = None):
        self.model = model
        self.input_shape = input_shape
        self._classes = None

    def fit(self, X, y, **kwargs):
        # Flatten if needed (for CNN/RNN inputs)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        # Convert one-hot to labels if needed
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        self._classes = np.unique(y)
        return self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict_proba(X, **kwargs)

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self


def create_svm(params: Dict[str, Any]) -> ClassicalModelWrapper:
    return ClassicalModelWrapper(SVC(**params))


def create_rf(params: Dict[str, Any]) -> ClassicalModelWrapper:
    return ClassicalModelWrapper(RandomForestClassifier(**params))


def create_knn(params: Dict[str, Any]) -> ClassicalModelWrapper:
    return ClassicalModelWrapper(KNeighborsClassifier(**params))


def create_lda(params: Dict[str, Any]) -> ClassicalModelWrapper:
    return ClassicalModelWrapper(LinearDiscriminantAnalysis(**params))


def create_nb(params: Dict[str, Any]) -> ClassicalModelWrapper:
    return ClassicalModelWrapper(GaussianNB(**params))


def create_mlp_sklearn(params: Dict[str, Any]) -> ClassicalModelWrapper:
    return ClassicalModelWrapper(MLPClassifier(**params))


CLASSICAL_MODELS = {
    'svm': create_svm,
    'rf': create_rf,
    'knn': create_knn,
    'lda': create_lda,
    'nb': create_nb,
    'mlp_sklearn': create_mlp_sklearn,
}

CLASSICAL_MODEL_NAMES = {
    'svm': 'Support Vector Machine',
    'rf': 'Random Forest',
    'knn': 'K-Nearest Neighbors',
    'lda': 'Linear Discriminant Analysis',
    'nb': 'Naive Bayes',
    'mlp_sklearn': 'MLP (sklearn)',
}