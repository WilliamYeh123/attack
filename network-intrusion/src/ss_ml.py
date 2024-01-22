####################################
# Author: S. A. Owerre
# Date modified: 12/03/2021
# Class: Semi-Supervised Learning
####################################

import warnings

warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


class SemiSupervised:
    """Semi-supervised learning classifiers."""

    def __init__(self):
        """Parameter initialization."""
        pass

    def self_training_clf(
        self,
        base_classifier,
        X_train,
        y_train,
        threshold=None,
        max_iter=None,
        verbose=None,
    ):
        """Train self-training classifier from scikit-learn >= 0.24.1.

        Parameters
        ----------
        base_classifier: supervised classifier implementing
        both fit and predict_proba
        X_train: scaled feature matrix of the training set
        y_train: class label of the training set
        threshold (float):  the decision threshold for
        use with criterion='threshold'. Should be in [0, 1)
        max_iter (int):  maximum number of iterations allowed.
        Should be greater than or equal to 0
        verbose (bool): enable verbose output

        Returns
        -------
        predicted labels and probability
        """
        # self training model
        model = SelfTrainingClassifier(
            base_classifier,
            threshold=threshold,
            max_iter=max_iter,
            verbose=verbose,
        )

        # fit the training set
        model.fit(X_train, y_train)

        # predict the labels of the unlabeled data points
        predicted_labels = model.predict(X_train)

        # predict probability
        predicted_proba = model.predict_proba(X_train)
        return predicted_labels, predicted_proba

    def label_spread(self, X_train, y_train, gamma=None, max_iter=None):
        """Train Label Spreading model from scikit-learn

        Parameters
        ----------
        X_train: scaled training data
        y_train: class label
        gamma: parameter for rbf kernel
        max_iter: maximum number of iterations allowed

        Returns
        -------
        Predicted labels and probability
        """
        # label spreading model
        model = LabelSpreading(
            kernel='rbf', gamma=gamma, max_iter=max_iter, n_jobs=-1
        )

        # fit the training set
        model.fit(X_train, y_train)

        # predict the labels of the unlabeled data points
        predicted_labels = model.transduction_

        # predict probability
        predicted_proba = model.predict_proba(X_train)
        return predicted_labels, predicted_proba

    def eval_metrics(self, y_true, y_pred, model_nm=None):
        """Evaluation metric using the ground truth and the predicted labels.

        Parameters
        ----------
        y_pred: predicted labels
        y_true: true labels
        model_nm: name of classifier

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

    def plot_varying_threshold(self, base_classifier, X_train, y_train):
        """Plot the effect of varying threshold for self-training.

        Parameters
        ----------
        base_classifier: supervised classifier implementing
        both fit and predict_proba
        X_train: scaled feature matrix of the training set
        y_train: class label of the training set

        Returns
        -------
        Matplotlib figure
        """
        total_samples = y_train.shape[0]
        x_values = np.arange(0.4, 1.05, 0.05)
        x_values = np.append(x_values, 0.99999)
        no_labeled = np.zeros(x_values.shape[0])
        no_iterations = np.zeros(x_values.shape[0])
        for (i, threshold) in enumerate(x_values):
            # fit model with chosen base classifier
            self_training_clf = SelfTrainingClassifier(
                base_classifier, threshold=threshold
            )
            self_training_clf.fit(X_train, y_train)

            # the number of labeled samples that the classifier
            # has available by the end of fit
            no_labeled[i] = (
                total_samples
                - np.unique(
                    self_training_clf.labeled_iter_, return_counts=True
                )[1][0]
            )

            # the last iteration the classifier labeled a sample in
            no_iterations[i] = np.max(self_training_clf.labeled_iter_)

        # plot figures
        plt.rcParams.update({'font.size': 15})
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
        ax1.plot(x_values, no_labeled, color='b')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Number of labeled samples')
        ax2.plot(x_values, no_iterations, color='b')
        ax2.set_ylabel('Number of iterations')
        ax2.set_xlabel('Threshold')
        plt.show()
