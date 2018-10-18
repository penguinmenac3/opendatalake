import pandas as pd

from opendatalake.simple_sequence import SimpleSequence


class CSV(SimpleSequence):
    def __init__(self, hyperparams, phase, preprocess_fn=None, augmentation_fn=None):
        super(CSV, self).__init__(hyperparams, phase, preprocess_fn, augmentation_fn)
        file_name = hyperparams.problem.filename
        feature_name_list = hyperparams.problem.feature_name_list
        label_name_list = hyperparams.problem.label_name_list
        df = pd.read_csv(file_name)

        self.features = df.loc[:, feature_name_list].values
        self.labels = df.loc[:, label_name_list].values

    def __num_samples(self):
        return len(features)

    def __get_sample(self, idx):
        return {"feature": features[idx]}, {"label": labels[idx]}


if __name__ == "__main__":
    print("Loading Dataset:")
    # Random csv-dataset from kaggle: https://www.kaggle.com/iliassekkaf/computerparts
    gen, params = load_csv("data/computer_parts/All_GPUs.csv", feature_name_list=["Manufacturer", "Memory"], label_name_list=["Name"])
    train_data = gen(params)

    feature, label = next(train_data)
    print(feature["feature"])
    print(label["label"])
