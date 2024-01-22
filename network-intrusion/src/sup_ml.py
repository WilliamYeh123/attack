##############################
# Author: S. A. Owerre
# Date modified: 12/03/2021
# Class: Supervised ML
##############################

import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, auc
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve


class SupervisedModels:
    """Model training of supervised ML."""

    def __init__(self):
        """Parameter initialization."""
        pass

    def prediction(self, model, X, y_true, subset=None, model_nm=None):
        """Predictions on the dataset.

        Parameters
        ----------
        model: trained supervised model
        X (array): feature matrix
        y_true (1d array): ground truth labels
        model_nm (str): name of classifier
        subset (str): subset of data

        Returns
        -------
        Performance metrics
        """
        # Make prediction
        y_pred = model.predict(X)

        # Compute the accuracy of the model
        accuracy = accuracy_score(y_true, y_pred)

        # Predict probability
        y_pred_proba = model.predict_proba(X)[:, 1]

        print(f'Predictions on the {subset} for {model_nm}')
        print('-' * 60)
        print('Accuracy:  %f' % (accuracy))
        print('AUROC: %f' % (roc_auc_score(y_true, y_pred_proba)))
        print('AUPRC: %f' % (average_precision_score(y_true, y_pred_proba)))
        print('Predicted classes:', np.unique(y_pred))
        print('Confusion matrix:\n', confusion_matrix(y_true, y_pred))
        print(
            'Classification report:\n', classification_report(y_true, y_pred)
        )
        print('-' * 60)

    def prediction_cv(
        self, model, X_train, y_train, cv_fold, scoring=None, model_nm=None
    ):
        """Cross-validation prediction on the training set.

        Parameters
        ----------
        model: supervised classification model
        X_train (array): feature matrix of the training set
        y_train (1d array): class labels
        cv_fold (int): number of cross-validation fold
        scoring (str): performance metric
        model_nm (str): name of classifier

        Returns
        -------
        Performance metrics on the cross-validation training set
        """

        # Fit the training set
        model.fit(X_train, y_train)

        # Compute accuracy on k-fold cross validation
        score = cross_val_score(
            model, X_train, y_train, cv=cv_fold, scoring=scoring
        )

        # Make prediction on k-fold cross validation
        y_cv_pred = cross_val_predict(model, X_train, y_train, cv=cv_fold)

        # Make probability prediction on k-fold cross validation
        y_pred_proba = cross_val_predict(
            model, X_train, y_train, cv=cv_fold, method='predict_proba'
        )[:, 1]

        # Print results
        print(
            f'{str(cv_fold)}-fold cross-validation for {str(model_nm)}',
            )
        print('-' * 60)
        print('Accuracy (std): %f (%f)' % (score.mean(), score.std()))
        print('AUROC: %f' % (roc_auc_score(y_train, y_pred_proba)))
        print('AUPRC: %f' % (average_precision_score(y_train, y_pred_proba)))
        print('Predicted classes:', np.unique(y_cv_pred))
        print('Confusion matrix:\n', confusion_matrix(y_train, y_cv_pred))
        print(
            'Classification report:\n',
            classification_report(y_train, y_cv_pred),
        )
        print('-' * 60)

    def plot_auc_ap_svm(self, X_train, y_train, cv_fold):
        """Plot of cross-validation AUC and AP for SVM.

        Parameters
        ----------
        X_train: feature matrix of the training set
        y_train: class labels
        cv_fold: number of cross-validation fold

        Returns
        ------
        matplolib figure of auc vs. hyperparameters
        """
        C_list = [2**x for x in range(-2, 9, 2)]
        gamma_list = [2**x for x in range(-11, -5, 2)]
        auc_list = [
            pd.Series(0.0, index=range(len(C_list)))
            for _ in range(len(gamma_list))
        ]
        ap_list = [
            pd.Series(0.0, index=range(len(C_list)))
            for _ in range(len(gamma_list))
        ]
        axes_labels = ['2^-2', '2^0', '2^2', '2^4', '2^6', '2^8']
        gamma_labels = ['2^-11', '2^-9', '2^-7']
        plt.rcParams.update({'font.size': 15})
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        for i, val1 in enumerate(gamma_list):
            for j, val2 in enumerate(C_list):
                model = SVC(
                    C=val2,
                    gamma=val1,
                    probability=True,
                    kernel='rbf',
                    random_state=42,
                )
                model.fit(X_train, y_train)
                y_pred_proba = cross_val_predict(
                    model, X_train, y_train, cv=cv_fold, method='predict_proba'
                )[:, 1]
                auc_list[i][j] = roc_auc_score(y_train, y_pred_proba)
                ap_list[i][j] = average_precision_score(y_train, y_pred_proba)
            auc_list[i].plot(
                label='gamma=' + str(gamma_labels[i]),
                marker='o',
                linestyle='-',
                ax=ax1,
            )
            ap_list[i].plot(
                label='gamma=' + str(gamma_labels[i]),
                marker='o',
                linestyle='-',
                ax=ax2,
            )

        ax1.set_xlabel('C', fontsize=15)
        ax1.set_ylabel('AUC', fontsize=15)
        ax1.set_title(
            f'{cv_fold}-Fold Cross-Validation with RBF Kernel SVM',
            fontsize=15,
        )
        ax1.set_xticklabels(axes_labels)
        ax1.set_xticks(range(len(C_list)))
        ax1.legend(loc='best')

        ax2.set_xlabel('C', fontsize=15)
        ax2.set_ylabel('AP', fontsize=15)
        ax2.set_title(
            f'{cv_fold}-Fold Cross-Validation with RBF Kernel SVM',
            fontsize=15,
        )
        ax2.set_xticks(range(len(C_list)))
        ax2.set_xticklabels(axes_labels)
        ax2.legend(loc='best')
        plt.show()

    def plot_auc_ap_lr(self, X_train, y_train, cv_fold=None):
        """Plot of cross-validation AUC and AP for Logistic regression.

        Parameters
        ----------
        X_train (array): feature matrix of the training set
        y_train (1d array): class labels
        cv_fold (int): number of cross-validation fold

        Returns
        -------
        matplolib figure of auc vs. hyperparameters
        """
        C_list = [2**x for x in range(-2, 9, 2)]
        class_wgt_list = [None, 'balanced', {0: 1, 1: 3}, {0: 1, 1: 61}]
        auc_list = [
            pd.Series(0.0, index=range(len(C_list)))
            for _ in range(len(class_wgt_list))
        ]
        ap_list = [
            pd.Series(0.0, index=range(len(C_list)))
            for _ in range(len(class_wgt_list))
        ]

        axes_labels = ['2^-2', '2^0', '2^2', '2^4', '2^6', '2^8']
        plt.rcParams.update({'font.size': 15})
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        for i, val1 in enumerate(class_wgt_list):
            for j, val2 in enumerate(C_list):
                model = LogisticRegression(
                    C=val2, class_weight=val1, random_state=42
                )
                model.fit(X_train, y_train)
                y_pred_proba = cross_val_predict(
                    model, X_train, y_train, cv=cv_fold, method='predict_proba'
                )[:, 1]
                auc_list[i][j] = roc_auc_score(y_train, y_pred_proba)
                ap_list[i][j] = average_precision_score(y_train, y_pred_proba)
            auc_list[i].plot(
                label='class_weight=' + str(class_wgt_list[i]),
                marker='o',
                linestyle='-',
                ax=ax1,
            )
            ap_list[i].plot(
                label='class_weight=' + str(class_wgt_list[i]),
                marker='o',
                linestyle='-',
                ax=ax2,
            )

        ax1.set_xlabel('C', fontsize=15)
        ax1.set_ylabel('AUC', fontsize=15)
        ax1.set_title(
            f'{cv_fold}-Fold Cross-Validation with Logistic Regression',
            fontsize=15,
        )
        ax1.set_xticklabels(axes_labels)
        ax1.set_xticks(range(len(C_list)))
        ax1.legend(loc='best')

        ax2.set_xlabel('C', fontsize=15)
        ax2.set_ylabel('AP', fontsize=15)
        ax2.set_title(
            f'{cv_fold}-Fold Cross-Validation with Logistic Regression',
            fontsize=15,
        )
        ax2.set_xticks(range(len(C_list)))
        ax2.set_xticklabels(axes_labels)
        ax2.legend(loc='best')
        plt.show()

    def plot_roc_pr_curves(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        cv_fold,
        color=None,
        label=None,
    ):
        """Plot ROC and PR curves for cross-validation and test sets.

        Parameters
        ----------
        model: trained supervised classification model
        X_train (array): feature matrix of the training set
        y_train (1d array): training set class labels
        X_test (array): feature matrix of the test set
        y_test (1d array): test set class labels
        cv_fold (int): number of k-fold cross-validation
        color (str): matplotlib color
        label (str): matplotlib label

        Returns
        -------
        Matplotlib line plot
        """

        # ROC and PR curves for cross-validation set

        # make prediction on k-fold cross validation
        y_cv_pred_proba = cross_val_predict(
            model, X_train, y_train, cv=cv_fold, method='predict_proba'
        )

        # compute the fpr and tpr for each classifier
        fpr_cv, tpr_cv, _ = roc_curve(y_train, y_cv_pred_proba[:, 1])

        # compute the precisions and recalls for the classifier
        precisions_cv, recalls_cv, _ = precision_recall_curve(
            y_train, y_cv_pred_proba[:, 1]
        )

        # compute the area under the ROC curve for each classifier
        area_auc_cv = roc_auc_score(y_train, y_cv_pred_proba[:, 1])

        # compute the area under the PR curve for the classifier
        area_prc_cv = auc(recalls_cv, precisions_cv)

        # ROC curve
        # plt.rcParams.update({'font.size': 12})
        plt.subplot(221)
        plt.plot(fpr_cv, tpr_cv, color=color, label=(label) % area_auc_cv)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False positive rate (FPR)')
        plt.ylabel('True positive rate (TPR)')
        plt.title(f'ROC Curve for the {cv_fold}-Cross-Validation Training Set')
        plt.legend(loc='best')

        # PR curve
        plt.subplot(223)
        plt.plot(
            recalls_cv, precisions_cv, color=color, label=(label) % area_prc_cv
        )
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(
            f'PR Curve for {cv_fold}-Fold Cross-Validation Training Set',
            )
        plt.legend(loc='best')

        # ROC and PR curves for Test set
        # predict probability
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # compute the fpr and tpr for each classifier
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

        # compute the precisions and recalls for the classifier
        precisions, recalls, _ = precision_recall_curve(y_test, y_pred_proba)

        # compute the area under the ROC curve for each classifier
        area_auc = roc_auc_score(y_test, y_pred_proba)

        # compute the area under the PR curve for the classifier
        area_prc = auc(recalls, precisions)

        # ROC curve
        # plt.rcParams.update({'font.size': 12})
        plt.subplot(222)
        plt.plot(fpr, tpr, color=color, label=(label) % area_auc)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False positive rate (FPR)')
        plt.ylabel('True positive rate (TPR)')
        plt.title('ROC Curve for the Test Set')
        plt.legend(loc='best')

        # PR curve
        plt.subplot(224)
        plt.plot(recalls, precisions, color=color, label=(label) % area_prc)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve for the Test Set')
        plt.legend(loc='best')

    def plot_aucroc_aucpr(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        cv_fold,
        marker=None,
        color=None,
        label=None,
    ):
        """Plot AUC-ROC  and AUC-PR curves for cross-validation vs. test sets.

        Parameters
        ----------
        model: trained supervised classification model
        X_train (array): feature matrix of the training set
        y_train (1d array): training set class labels
        X_test (array): feature matrix of the test set
        y_test (1d array): test set class labels
        cv_fold (int): number of k-fold cross-validation
        color (str): matplotlib color
        marker (str): matplotlib marker

        Returns
        -------
        Matplotlib line plot
        """

        # AUC-ROC  and AUC-PR for Test set

        # predict probability on the test set
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # compute the precisions and recalls of the test set
        test_precisions, test_recalls, _ = precision_recall_curve(
            y_test, y_pred_proba
        )

        # compute the area under the ROC curve of the test set
        area_auc_test = roc_auc_score(y_test, y_pred_proba)

        # compute the area under the PR curve on the test set
        area_prc_test = auc(test_recalls, test_precisions)

        # AUC-ROC  and AUC-PR cross-validation training set

        # make prediction on the k-fold cross-validation set
        y_cv_pred_proba = cross_val_predict(
            model, X_train, y_train, cv=cv_fold, method='predict_proba'
        )

        # compute the precisions and recalls of the cross-validation set
        cv_precisions, cv_recalls, _ = precision_recall_curve(
            y_train, y_cv_pred_proba[:, 1]
        )

        # compute the area under the ROC curve of the cross-validation set
        area_auc_cv = roc_auc_score(y_train, y_cv_pred_proba[:, 1])

        # compute the area under the PR curve of the cross-validation set
        area_prc_cv = auc(cv_recalls, cv_precisions)

        # plot
        # AUC-ROC
        plt.subplot(121)
        plt.plot(
            [area_auc_cv],
            [area_auc_test],
            color=color,
            marker=marker,
            label=label,
        )
        plt.plot([0.979, 1.001], [0.979, 1.001], 'k--', linewidth=0.5)
        plt.axis([0.979, 1.001, 0.979, 1.001])
        plt.xticks(np.arange(0.98, 1, 0.01))
        plt.yticks(np.arange(0.98, 1, 0.01))
        plt.xlabel('Cross-validation set results')
        plt.ylabel('Test set results')
        plt.title('AUC-ROC for Cross-Validation vs. Test Sets')
        plt.legend(loc='best')

        # AUC-PR
        plt.subplot(122)
        plt.plot(
            [area_prc_cv],
            [area_prc_test],
            color=color,
            marker=marker,
            label=label,
        )
        plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
        plt.axis([0.87, 1.001, 0.87, 1.001])
        plt.xticks(np.arange(0.88, 1.01, 0.02))
        plt.yticks(np.arange(0.88, 1.02, 0.02))
        plt.xlabel('Cross-validation set results')
        plt.ylabel('Test set results')
        plt.title('AUC-PR for Cross-Validation vs. Test Sets')
        plt.legend(loc='best')
