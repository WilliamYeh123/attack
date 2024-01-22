##################################
# Author: S. A. Owerre
# Date modified: 12/03/2021
# Class: Transformation Pipeline
##################################

import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


class TransformationPipeline:
    """Transformation pipeline for semi-supervised learning."""

    def __init__(self):
        """Define parameters."""
        pass

    def num_pipeline(self, X_train):
        """Transformation pipeline of data with only numerical variables.

        Parameters
        ----------
        X_train: training set

        Returns
        -------
        Transformation pipeline and transformed data in numpy array
        """
        # original numerical feature names
        feat_names = list(X_train.select_dtypes('number'))

        # create pipeline
        num_pipeline = Pipeline(
            [
                ('std_scaler', StandardScaler()),
            ]
        )
        # apply transformer
        X_train_scaled = num_pipeline.fit_transform(X_train)
        return X_train_scaled, feat_names

    def cat_encoder(self, X_train):
        """Encoder for categorical variables.

        Parameters
        ----------
        X_train: training set

        Returns
        -------
        Transformation pipeline and transformed data in array
        """
        # Instatiate class
        one_hot_encoder = OneHotEncoder()

        # Fit transform the training set
        X_train_scaled = one_hot_encoder.fit_transform(X_train)

        # Feature names for output features
        feat_names = list(
            one_hot_encoder.get_feature_names_out(
                list(X_train.select_dtypes('O'))
            )
        )
        return X_train_scaled.toarray(), feat_names

    def preprocessing(self, X_train):
        """Transformation pipeline of data with both numerical
        and categorical variables.

        Parameters
        ----------
        X_train: training set

        Returns
        -------
        Transformed data in array
        """

        # numerical transformation pipepline
        num_train, num_col = self.num_pipeline(X_train.select_dtypes('number'))

        # categorical transformation pipepline
        cat_train, cat_col = self.cat_encoder(X_train.select_dtypes('O'))

        # transformed training set
        X_train_scaled = np.concatenate((num_train, cat_train), axis=1)

        # feature names
        feat_names = num_col + cat_col
        return X_train_scaled, feat_names

    def pca_plot_labeled(self, X, labels, palette=None, ax=None):
        """Dimensionality reduction of labeled data using PCA.

        Parameters
        ----------
        X: transformed and scaled data
        labels: class labels
        palette: color list
        ax: matplotlib axes

        Returns
        -------
        Matplotlib plot of two component PCA
        """
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # put in dataframe
        X_reduced_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
        X_reduced_pca['class'] = labels

        # plot results
        sns.scatterplot(
            x='PC1',
            y='PC2',
            data=X_reduced_pca,
            hue='class',
            style='class',
            palette=palette,
            ax=ax,
        )

        # axis labels
        ax.set_xlabel('Principal component 1')
        ax.set_ylabel('Principal component 2')
        ax.legend(loc='best')

    def plot_pca(self, X_train, y_train, y_pred):
        """Plot PCA before and after semi-supervised classification.

        Parameters
        ----------
        X_train: scaled feature matrix of the training set
        y_train: original labels of the training set
        y_pred: predicted labels of the unlabeled data points

        Returns
        -------
        Matplotlib figure
        """
        # Plot figure
        plt.rcParams.update({'font.size': 15})
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        self.pca_plot_labeled(
            X_train, y_train, palette=['gray', 'lime', 'r'], ax=ax1
        )
        self.pca_plot_labeled(X_train, y_pred, palette=['lime', 'r'], ax=ax2)
        ax1.set_title('PCA before semi-supervised classification')
        ax2.set_title('PCA after semi-supervised classification')
