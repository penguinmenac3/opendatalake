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

    def num_samples(self):
        return len(self.features)

    def get_sample(self, idx):
        return {"feature": self.features[idx]}, {"label": self.labels[idx]}
