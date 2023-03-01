import pickle
from MLClass import MachineLearning
from sklearn.neighbors import KNeighborsClassifier


class in_session_KNN(MachineLearning):
    """
    Class implementing the knn model. Inherits ML functions from MachineLearning Class.
    """

    def __init__(self):
        super().__init__()

    def loop_matrices(self, source_path, n_neighbors,weights):
        """
        Method loops through matrices, prepares training and test data set and fits classifier. 
        Calls function to calculate performance metrics and saves them.
        :param source_path: source path were data is stored
        :param n_neighbors: number of neighbors
        :param weights: weights function use in prediction 
        :return: metrics
        """
        for i in self.range_n:
            df = self.load_data(i, source_path)
            cv, X_train, X_test, y_train, y_test = self.prepare_training_set(df)

            # fit
            knn = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights)
            knn = knn.fit(X_train, y_train)
            pred = knn.predict(X_test)

            # call function to get metrics
            a, p, r, roc_auc, fpr = self.get_metrics(y_test, pred)

            if(self.minority_group !=None):
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
            else:
                self.metrics = self.non_aggregated(a, p, r, roc_auc, fpr,i,df )

        return self.metrics
    
    def non_aggregated(self,a, p, r, roc_auc, fpr,i,df):
        """
        add accuracy metrics for the group in aggregation bias mitigation
        :param a,p,r,roc_auc,fpr: metrics
        :param i: range
        :param df: dataframe
        :return: metrics
        """
        self.metrics = self.metrics.append(
        {
            "model": "KNN",
            "group": self.demographic_category,
            "subgroup": self.majority_group,
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

        return self.metrics
        
        
