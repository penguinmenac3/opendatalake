import pandas as pd


def _gen(params, stride=1, offset=0, infinite=False):
    features, labels, prepare_features, prepare_labels = params

    loop_condition = True
    while loop_condition:
        for idx in range(offset, len(features), stride):
            feature = features[idx]
            if prepare_features:
                feature = prepare_features(feature)
            label = labels[idx]
            if prepare_labels:
                label = prepare_labels(label)
            yield ({"feature": feature}, {"label": label})
        loop_condition = infinite

def load_csv(file_name, feature_name_list=[], label_name_list=[], prepare_features=None, prepare_labels=None):
    df = pd.read_csv(file_name)

    features = df.loc[:, feature_name_list].values
    labels = df.loc[:, label_name_list].values

    return _gen, (features, labels, prepare_features, prepare_labels)


if __name__ == "__main__":
    print("Loading Dataset:")
    # Random csv-dataset from kaggle: https://www.kaggle.com/iliassekkaf/computerparts
    gen, params = load_csv("data/computer_parts/All_GPUs.csv", feature_name_list=["Manufacturer", "Memory"], label_name_list=["Name"])
    train_data = gen(params)

    feature, label = next(train_data)
    print(feature["feature"])
    print(label["label"])
