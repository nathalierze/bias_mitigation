import pickle
from MLClass import MachineLearning


class in_session_MLP(MachineLearning):
    """
    Describe class
    """

    def __init__(self):
        super().__init__()

    def noname(self):
        for i in n:
            path = "gender_historical/matrix" + str(i) + ".pkl"
            infile = open(path, "rb")
            df = pickle.load(infile)
            infile.close()
            df = df.reset_index()

            # prepare features
            y_len = len(feature_cols)
            X = df[feature_cols].astype(float)
            y = df.y
            y = y.astype("int")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=1
            )

            model = build_model()

            model.compile(
                loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"]
            )

            model.fit(
                x=X_train,
                y=y_train,
                epochs=10,
                batch_size=128,
                verbose=0,
                validation_data=(X_test, y_test),
            )

            scores = model.evaluate(x=X_test, y=y_test, verbose=0)

            # call function to get metrics and append metrics to df
            a, p, r, roc_auc, fpr = get_dn_metrics(model, X_test, y_test)
            metrics = metrics.append(
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

            ##
            # let the model above predict for each subgroup and save results to evaluate later
            group = ["gender", "gender"]
            subgroup = ["boys", "girls"]
            matrice = ["matrices_forte_boys", "matrices_forte_girls"]

            for group, subgroup, matrix in zip(group, subgroup, matrice):
                path = (
                    "../../02_dropout_prediction/01_keep_it_up/"
                    + matrix
                    + "/matrix"
                    + str(i)
                    + ".pkl"
                )
                infile = open(path, "rb")
                df = pickle.load(infile)
                infile.close()
                df = df.reset_index()
                y_len = len(feature_cols)
                X = df[feature_cols].astype(float)
                y = df.y
                y = y.astype("int")

                # call function to get metrics and append to df
                a, p, r, roc_auc, fpr = get_dn_metrics(model, X, y)
                metrics = metrics.append(
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

        def build_model():
            """ "
            build dropout prediction model
            """
            model = Sequential()
            model.add(Dense(22, input_dim=22, activation="relu"))
            model.add(Dense(44, activation="relu"))
            model.add(Dense(22, activation="relu"))
            model.add(Dense(11, activation="relu"))
            model.add(Dense(1, activation="sigmoid"))

            return model

        def get_dn_metrics(model, X, y):
            """
            calculate and extract relevant metrics from y and pred
            return metrics
            """
            yhat_probs = model.predict(X, verbose=0)
            yhat_classes = (model.predict(X) > 0.5).astype("int32")
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
