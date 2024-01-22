##############################
# Author: S. A. Owerre
# Date modified: 12/03/2021
# Class: Unsupervised ML
##############################

import warnings

warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.cblof import CBLOF
from pyod.models.pca import PCA as PCAOD
from sklearn.svm import OneClassSVM as OCSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import average_precision_score


class UnsupervisedModels:
    """Unsupervised ML models."""

    def __init__(self):
        """Parameter initialization."""
        pass

    def iforest(self, X_train, n_estimators=None, random_state=None):
        """Train Isolation Forest from scikit-learn.

        Parameters
        ----------
        X_train: scaled training data
        n_estimators: number of isolation trees
        random_state: random number seed

        Returns
        -------
        Anomaly scores
        """
        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples='auto',
            random_state=random_state,
        )
        model.fit(X_train)

        # predict raw anomaly score
        labels = model.predict(X_train)   # -1 for outliers and 1 for inliers
        labels = (labels.max() - labels) // 2   # 1: outliers, 0: inliers
        iforest_anomaly_scores = model.decision_function(X_train) * -1
        iforest_anomaly_scores = self.min_max_scaler(iforest_anomaly_scores)
        return iforest_anomaly_scores, labels

    def cblof(self, X_train, contamination=None, random_state=None):
        """Train CBLOF model from PYOD.

        Parameters
        ----------
        X_train: scaled training data
        contamination: percentage of anomalies in the data
        random_state: random number seed

        Returns
        -------
        Anomaly scores
        """
        model = CBLOF(contamination=contamination, random_state=random_state)
        model.fit(X_train)

        # predict raw anomaly score
        labels = model.predict(X_train)   # outlier labels (0 or 1)
        cblof_anomaly_scores = model.decision_function(X_train)
        cblof_anomaly_scores = self.min_max_scaler(cblof_anomaly_scores)
        return cblof_anomaly_scores, labels

    def ocsvm(self, X_train, kernel=None, gamma=None, nu=None):
        """Train OCSVM model from Sklearn.

        Parameters
        ----------
        X_train: scaled training data
        kernel: kernel funcs: linear, poly, rbf, sigmoid
        gamma: kernel coefficient for rbf, poly and sigmoid
        nu: regularization parameter btw [0,1]

        Returns
        -------
        Anomaly scores
        """
        model = OCSVM(kernel=kernel, gamma=gamma, nu=nu)
        model.fit(X_train)

        # predict raw anomaly score
        labels = model.predict(X_train)  # Outlier labels (-1 or 1)
        labels = (labels.max() - labels) // 2   # 1: outliers, 0: inliers
        ocsvm_anomaly_scores = model.decision_function(X_train) * -1
        ocsvm_anomaly_scores = self.min_max_scaler(ocsvm_anomaly_scores)
        return ocsvm_anomaly_scores, labels

    def cov(self, X_train, contamination=None, random_state=None):
        """Train Elliptic Envelope model from scikit-learn.

        Parameters
        ----------
        X_train: scaled training data
        contamination: percentage of anomalies in the data
        random_state: random number seed

        Returns
        -------
        Anomaly scores
        """
        model = EllipticEnvelope(
            contamination=contamination, random_state=random_state
        )
        model.fit(X_train)

        # predict raw anomaly score
        labels = model.predict(X_train)   # -1 for outliers and 1 for inliers
        labels = (labels.max() - labels) // 2   # 1: outliers, 0: inliers)
        cov_anomaly_scores = model.decision_function(X_train) * -1
        cov_anomaly_scores = self.min_max_scaler(cov_anomaly_scores)
        return cov_anomaly_scores, labels

    def pca(self, X_train, n_components=None, contamination=None):
        """Train PCA model from PYOD.

        Parameters
        ----------
        X_train: scaled training data
        contamination: percentage of anomalies in the data
        n_components: number of components to transform

        Returns
        -------
        Anomaly scores
        """
        model = PCAOD(n_components=n_components, contamination=contamination)
        model.fit(X_train)

        # predict raw anomaly score
        labels = model.predict(X_train)  # outlier labels (0 or 1)
        pca_anomaly_scores = model.decision_function(
            X_train
        )   # outlier scores
        pca_anomaly_scores = self.min_max_scaler(pca_anomaly_scores)
        return pca_anomaly_scores, labels

    def eval_metric(self, y_true, y_pred, model_nm=None):
        """Evaluation metric using the ground truth and the predicted labels.

        Parameters
        ----------
        y_pred: predicted labels
        y_true: true labels
        model_nm: name of model

        Returns
        -------
        Performance metrics
        """
        print(f'Test predictions for {str(model_nm)}')
        print('-' * 60)
        print('Accuracy:  %f' % (accuracy_score(y_true, y_pred)))
        print('AUROC: %f' % (roc_auc_score(y_true, y_pred)))
        print('AUPRC: %f' % (average_precision_score(y_true, y_pred)))
        print('Predicted classes:', np.unique(y_pred))
        print('Confusion matrix:\n', confusion_matrix(y_true, y_pred))
        print(
            'Classification report:\n', classification_report(y_true, y_pred)
        )
        print('-' * 60)

    def min_max_scaler(self, arr):
        """Min-Max normalization to rescale the anomaly scores.

        Parameters
        ----------
        arr: 1d array

        Returns
        -------
        normalized array in the range [0,100]
        """
        scaler = (arr - np.min(arr)) * 100 / (np.max(arr) - np.min(arr))
        return scaler

    def plot_dist(self, scores, color=None, title=None):
        """Plot the distribution of anomaly scores.

        Parameters
        ----------
        scores: scaled anomaly scores

        Returns
        -------
        seaborn distribution plot
        """
        # figure layout
        plt.rcParams.update({'font.size': 15})
        plt.subplots(figsize=(8, 6))

        # plot distribution with seaborn
        sns.distplot(scores, color=color)
        plt.title(label=title)
        plt.xlabel('Normalized anomaly scores')
        plt.show()
