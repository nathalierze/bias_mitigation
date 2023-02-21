import pickle
import pandas as pd
from imblearn.over_sampling import SMOTE


class Mitigation:
    """
    This class creates the data object to  mitigate and splits it into matrices for temporal prediction models
    """

    path_ = "c:/Users/Nathalie/Nextcloud/LADi/Orthografie Trainer/Code/"
    mitigation_path = "04_bias_mitigation/"

    def __init__(self):
        self.range_n = None
        self.minority_group = None
        self.majority_group = None
        self.demographic_category = None

        print("Created a Mitigation Object")

    def set_range(self, temp_start, temp_end):
        self.range_n = list(range(temp_start, temp_end))

    def set_minority_group(self, value):
        self.minority_group = value

    def set_majority_group(self, value):
        self.majority_group = value

    def set_demographic_category(self, value):
        self.demographic_category = value

    def get_range_n(self):
        return self.range_n

    def get_minority_group(self):
        return self.minority_group

    def get_majority_group(self):
        return self.majority_group

    def get_demographic_category(self):
        return self.demographic_category

    def load_matrices(self, folder, file_type, balancing):
        """
        Method loads data per matrix and only keeps majority and minority group
        """
        for i in self.range_n:
            path = (
                self.path_
                + "02_dropout_prediction/01_keep_it_up/matrices_allsessions/matrix"
                + str(i)
                + ".pkl"
            )
            infile = open(path, "rb")
            df = pickle.load(infile)
            infile.close()
            df = df.reset_index(level=0)

            df_1 = df[df[self.majority_group] == 1]
            if self.minority_group == None:
                df_0 = df[df[self.majority_group] == 0]
            else:
                df_0 = df[df[self.minority_group] == 1]
            df = pd.concat([df_0, df_1])

            if balancing == True:
                df = self.oversampling_minority(df)

            path = self.path_ + self.mitigation_path + folder + str(i) + file_type
            df.to_pickle(path)

    def oversampling_minority(self, df):
        X_df = df.drop(columns=[self.majority_group])
        y_df = df[self.majority_group]
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_df, y_df)
        df = X_train_smote.join(
            pd.DataFrame(list(y_train_smote.values), columns=[self.majority_group])
        )

        return df
