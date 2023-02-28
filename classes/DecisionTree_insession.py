import pickle
from MLClass import MachineLearning
from sklearn.tree import DecisionTreeClassifier


class in_session_decision_tree(MachineLearning):
    """
    Class implementing the decision tree model. Inherits ML functions from MachineLearning Class.
    """

    def __init__(self):
        super().__init__()

    def loop_matrices(self, source_path,max_depth, min_samples_leaf, min_samples_split):
        """
        Method loops through matrices, prepares training and test data set and fits classifier. 
        Calls function to calculate performance metrics and saves them.
        :param source_path: source path were data is stored
        :param max_depth: defines the maximum depth of decision tree
        :param min_samples_leaf: defines minimum number of samples needed to be at a leaf node
        :param min_samples_split: defines minimum number of samples needed to split a node  
        :return: metrics
        """
        for i in self.range_n:
            df = self.load_data(i, source_path)
            cv, X_train, X_test, y_train, y_test = self.prepare_training_set(df)

            # fit
            clf = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split)
            clf = clf.fit(X_train, y_train)
            pred = clf.predict(X_test)

            # call function to get metrics
            a, p, r, roc_auc, fpr = self.get_metrics(y_test, pred)

            # append metrics to df
            self.metrics = self.metrics.append(
                {
                    "model": "DTE",
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
            self.metrics = self.predict_subgroups(i, clf, "DTE")

        return self.metrics
