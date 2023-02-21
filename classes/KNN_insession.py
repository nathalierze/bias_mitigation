import pickle
from MLClass import MachineLearning
from sklearn.neighbors import KNeighborsClassifier


class in_session_KNN(MachineLearning):
    """
    Describe class
    """

    def __init__(self):
        super().__init__()

    def loop_matrices(self, source_path):
        for i in self.range_n:
            df = self.load_data(i, source_path)
            cv, X_train, X_test, y_train, y_test = self.prepare_training_set(df)

            # fit
            knn = KNeighborsClassifier(n_neighbors=2)
            knn = knn.fit(X_train, y_train)
            pred = knn.predict(X_test)

            # call function to get metrics
            a, p, r, roc_auc, fpr = self.get_metrics(y_test, pred)

            # append metrics to df
            self.metrics = self.metrics.append(
                {
                    "model": "KNN",
                    "group": "all",
                    "subgroup": "all",
                    "Length": len(df),
                    "Sentence": i,
                    "Accuracy": a,
                    "Precision": p,
                    "Recall": r,
                    "AUC": roc_auc,
                    "FPR": fpr,
                },
                ignore_index=True,
            )
            self.metrics = self.predict_subgroups(i, knn, "KNN")

        return self.metrics
