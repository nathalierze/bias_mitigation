from MitigationObjectClass import Mitigation
import pandas as pd

class Evaluation(Mitigation):
    """
    Describe class
    """

    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics

    def evaluate(self):
        
        grouped = self.metrics.groupby(self.metrics.group)
        df = grouped.get_group(self.demographic_category)
       
        df = df.drop(columns=["group", "Accuracy"])
        df = pd.pivot_table(
            df,
            values=["Precision", "Recall", "AUC", "FPR"],
            index=["model", "Sentence"],
            columns=["subgroup"],
        )
        
        df["PP"] = df.Precision[self.minority_group] - df.Precision[self.majority_group]
        df["EO"] = df.Recall[self.majority_group] - df.Recall[self.minority_group]
        df["SA"] = df.AUC[self.minority_group] - df.AUC[self.majority_group]
        df["PE"] = df.FPR[self.majority_group] - df.FPR[self.minority_group]
        df = df.drop(columns=["AUC", "Precision", "Recall", "FPR"])
        df.columns = df.columns.droplevel(1)
        df = pd.pivot_table(
            df, values=["PP", "EO", "SA", "PE"], index=["Sentence"], columns=["model"]
        )

        df = self.create_table(df)

        return df

    def create_table(self,df):
        met = ["EO", "PE", "PP", "SA"]
        model = ["DL", "DTE", "KNN"]
        ranges = [
            ("02-9", 8, 2, 10),
            ("10-19", 9, 10, 20),
            ("20-29", 9, 20, 30),
            ("30-39", 9, 30, 40),
            ("40-49", 9, 40, 50),
            ("50-60", 10, 50, 60),
        ]

        frame_means = pd.DataFrame()

        # for each metric
        for m in met:
            for mo in model:
                for r, div, begin, end in ranges:
                    s = 0
                    for i in range(begin, end):
                        s += df[m][mo][i]
                    temp = pd.DataFrame(
                        {"Metrik": [m], "Model": mo, "Range": r, "Val": s / div}
                    )
                    frame_means = pd.concat([frame_means, temp])

        # pivot table
        mean_table = pd.pivot_table(
            frame_means, values=["Val"], index=["Range"], columns=["Metrik", "Model"]
        )
        return mean_table

    def threshold001(self,v, props=""):
        return props if (v > 0.02) or (v < -0.02) else None


    def threshold005(self,v, props=""):
        return props if (v > 0.05) or (v < -0.05) else None


    def negativeValue(self,v, props=""):
        return props if (v < 0) else None

    def showTable(self,df):
        """
        functions to format results
        set two threshols: one at |0.02| in orange and one at |0.05| in red
        format all negative values in bold
        """
        styled = (
            df.style.set_properties(color="black", align="right")
            .set_properties(**{"background-color": "white"})
            .applymap(self.threshold001, props="color:orange;")
            .applymap(self.threshold005, props="color:red;")
            .applymap(self.negativeValue, props="font-weight:bold;")
        )
        return styled