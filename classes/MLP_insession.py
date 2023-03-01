import pickle
from MLClass import MachineLearning
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

class in_session_MLP(MachineLearning):
    """
    Class builds and fits MLP model
    """

    def __init__(self):
        super().__init__()
        self.model = None

    def loop_matrices(self, source_path, optimizer, loss, mlp_metric,input_dim, nodes):
        """
        Method loops through matrices, prepares training and test data set and fits model. 
        Calls function to calculate performance metrics and saves them.
        :param source_path: source path were data is stored
        :param optimizer, loss, mlp_metric: model parameters
        :param input_dim: defines input_dim of input layer
        :param nodes: defines nodes of input layer
        :return: metrics
        """
        for i in self.range_n:
            df = self.load_data(i, source_path)

            X = df[self.feature_cols].astype(float)
            y = df.y
            y = y.astype("int")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=1
            )

            self.model = self.build_model(input_dim, nodes)

            self.model.compile(
                loss=loss, optimizer=optimizer, metrics=[mlp_metric]
            )

            self.model.fit(
                x=X_train,
                y=y_train,
                epochs=10,
                batch_size=128,
                verbose=0,
                validation_data=(X_test, y_test),
            )

            scores = self.model.evaluate(x=X_test, y=y_test, verbose=0)

            # call function to get metrics and append metrics to df
            a, p, r, roc_auc, fpr = self.get_mlp_metrics(X_test, y_test)
            
            self.metrics = self.metrics.append(
                {
                    "model": "DL",
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

            self.metrics = self.predict_subgroups(i)

        return self.metrics


    def build_model(self,input_dim, nodes):
        """
        build mlp model
        :param input_dim: defines input_dim of input layer
        :param nodes: defines nodes of input layer
        :return: model
        """
        self.model = Sequential()
        self.model.add(Dense(nodes, input_dim=input_dim, activation="relu"))
        self.model.add(Dense(44, activation="relu"))
        self.model.add(Dense(nodes, activation="relu"))
        self.model.add(Dense(11, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))

        return self.model

    def get_mlp_metrics(self, X, y):
        """
        calculate and extract relevant metrics
        :param X,y: x and y 
        :return: a,p,r,roc_auc,fpr model metrics
        """
        yhat_probs = self.model.predict(X, verbose=0)
        yhat_classes = (self.model.predict(X) > 0.5).astype("int32")
        # reduce to 1d array
        yhat_probs = yhat_probs[:, 0]
        yhat_classes = yhat_classes[:, 0]
        a = accuracy_score(y, yhat_classes)
        p = precision_score(y, yhat_classes)
        r = recall_score(y, yhat_classes)
        roc_auc = roc_auc_score(y, yhat_probs)
        tn, fp, fn, tp = confusion_matrix(y, yhat_classes).ravel()
        fpr = fp / (fp + tn)

        return a, p, r, roc_auc, fpr

    def predict_subgroups(self, i):
        """
        Predicts metrics for each sub group with model
        :param i: sentence number

        :return: metrics
        """
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
            X = df[self.feature_cols].astype(float)
            y = df.y
            y = y.astype("int")

            a, p, r, roc_auc, fpr = self.get_mlp_metrics(X, y)


            if(self.minority_group !=None):
                self.metrics = self.metrics.append(
                    {
                        "model": "DL",
                        "group": group,
                        "subgroup": subgroup,
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
            "model": "DL",
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
        