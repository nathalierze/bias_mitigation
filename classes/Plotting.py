import pandas as pd
import pickle
import seaborn as sns

class Plots:
    def __init__(self):
        pass


    def plot_count_of_sentences_per_matrix(self, n_start,n_end,source_path):
        sentence_len = pd.DataFrame(columns=["Sentence", "Count"])
        n= list(range(n_start, n_end))
        for x in n:
            path = source_path + str(x) + ".pkl"
            infile = open(path, "rb")
            get_length = pickle.load(infile)
            infile.close()
            l = len(get_length)
            sentence_len = sentence_len.append({"Sentence": x, "Count": l}, ignore_index=True)

        sentence_len["Sentence"] = sentence_len["Sentence"].astype("int")
        sentence_len["Count"] = sentence_len["Count"].astype("int")
        sns.lineplot(data=sentence_len, x="Sentence", y="Count")

