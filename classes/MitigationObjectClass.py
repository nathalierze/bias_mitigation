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

        :param folder: specifies folder to store result
        :param file_type: specifies file type of the result
        :param balancing: binary indicates if an oversampling should be carried out
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

            if self.demographic_category == "AbiEltern":
                df = self.prepare_feature_Abi_Eltern(df)
            elif self.demographic_category == "Buecher":
                df = self.prepare_feature_buecher(df)
            elif self.demographic_category == 'eigSprache':
                df = self.prepare_feature_eig_sprache(df)
            else: 
                df = self.prepare_feature(df)

            if balancing == True:
                df = self.oversampling_minority(df)

            path = self.path_ + self.mitigation_path + folder + str(i) + file_type
            df.to_pickle(path)

    def prepare_feature_Abi_Eltern (self,df):
        df = self.add_survey_data(df)
        df[self.demographic_category] = df[self.demographic_category].astype("float")
        df[self.demographic_category] = df[self.demographic_category].replace([2], 1)
        df_1 = df[df[self.demographic_category] == 1]
        df_0 = df[df[self.demographic_category] == 0]
        df = pd.concat([df_0, df_1])

        return df

    def prepare_feature_eig_sprache (self,df):
        df = self.add_survey_data(df)
        df_1 = df[df[self.demographic_category] == 1]
        df_0 = df[df[self.demographic_category] == 0]
        df = pd.concat([df_0, df_1])

        return df

    def prepare_feature (self,df):
        df_1 = df[df[self.majority_group] == 1]
        df_0 = df[df[self.minority_group] == 1]
        df = pd.concat([df_0, df_1])

        return df

    def prepare_feature_buecher (self,df):
        df = self.add_survey_data(df)
        df[self.demographic_category] = df[self.demographic_category].replace(["10"], 0)
        df[self.demographic_category] = df[self.demographic_category].replace(["200"], 1)
        df_0 = df[df[self.demographic_category] == 0.0]
        df_1 = df[df[self.demographic_category] == 1]
        df = pd.concat([df_0, df_1])
        df[self.demographic_category] = df[self.demographic_category].astype("float")

        return df

    def add_survey_data(self, df):
        infile = open("../../02_dropout_prediction/01_keep_it_up/fairness_ready.pkl", "rb")
        survey_data = pickle.load(infile)
        infile.close()

        survey_data = survey_data[["UebungsID", self.demographic_category]]
        survey_data = survey_data.drop_duplicates()
        df = pd.merge(df, survey_data, how="left", on="UebungsID")

        return df
        
    def oversampling_minority(self, df):
        """
        method to oversample minority groups

        :param df: data frame to with two unbalanced groups
        :return: df with resampled groups
        """
        X_df = df.drop(columns=[self.majority_group])
        y_df = df[self.majority_group]
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_df, y_df)
        df = X_train_smote.join(
            pd.DataFrame(list(y_train_smote.values), columns=[self.majority_group])
        )

        return df
