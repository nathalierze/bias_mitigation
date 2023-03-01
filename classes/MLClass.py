import pandas as pd
import pickle
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, KFold
from MitigationObjectClass import Mitigation


class MachineLearning(Mitigation):
    """
    Machine Learning Class that provides functions for different ml implementations
    """

    path_ = "c:/Users/Nathalie/Nextcloud/LADi/Orthografie Trainer/Code/"

    def __init__(self):
        super().__init__()
        self.feature_cols = None
        self.metrics = pd.DataFrame(
            columns=[
                "model",
                "group",
                "subgroup",
                "Length",
                "Sentence",
                "Accuracy",
                "Precision",
                "Recall",
                "AUC",
                "FPR"
            ]
        )

        print("Hi ML class")

    def set_feature_cols(self, list):
        self.feature_cols = list

    def get_feature_cols(self):
        return self.feature_cols

    def get_metric_df(self):
        return self.metrics

    def load_data(self, i, source_path):
        """
        load data matrix from specific source
        :param source_path: specifies path of the matrix
        :return: loaded data as df
        """
        path = source_path + str(i) + ".pkl"
        infile = open(path, "rb")
        df = pickle.load(infile)
        infile.close()
        df = df.reset_index()

        return df

    def prepare_training_set(self, df):
        """
        prepares given df for model fitting
        :param df: df to be prepared
        :return: training and test set
        """
        # prepare features
        X = df[self.feature_cols]
        y = df.y
        y = y.astype("int")

        # prepare training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1
        )
        k = 5
        cv = KFold(n_splits=k, random_state=None)

        return cv, X_train, X_test, y_train, y_test

    def get_metrics(self, y, pred):
        """
        calculate and extract relevant metrics from y and pred
        return metrics
        """
        a = accuracy_score(y, pred)
        p = precision_score(y, pred)
        r = recall_score(y, pred)
        roc_auc = roc_auc_score(y, pred)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        fpr = fp / (fp + tn)

        return a, p, r, roc_auc, fpr

    def predict_subgroups(self, i, clf, model):
        group = [self.demographic_category, self.demographic_category]
        subgroup = [self.majority_group, self.minority_group]
        matrice = [
            "matrices_forte_" + self.majority_group,
            "matrices_forte_" + self.minority_group,
        ]

        for group, subgroup, matrix in zip(group, subgroup, matrice):
            path = (
                self.path_
                + "04_bias_mitigation/00_data/"
                + matrix
                + "/matrix"
                + str(i)
                + ".pkl"
            )
            infile = open(path, "rb")
            df = pickle.load(infile)
            infile.close()
            df = df.reset_index()
            X = df[self.feature_cols]
            y = df.y
            y = y.astype("int")
            pred = clf.predict(X)

            # call function to get metrics
            a, p, r, roc_auc, fpr = self.get_metrics(y, pred)
            self.metrics = self.metrics.append(
                {
                    "model": model,
                    "group": group,
                    "subgroup": subgroup,
                    "Length": len(df),
                    "Sentence": i,
                    "Accuracy": a,
                    "Precision": p,
                    "Recall": r,
                    "AUC": roc_auc,
                    "FPR": fpr
                },
                ignore_index=True,
            )

        return self.metrics
