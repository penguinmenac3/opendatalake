import os
import pickle
import numpy as np


# Here the cifar data can be downloaded.
CIFAR_10_DOWNLOAD = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_100_DOWNLOAD = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
# Extract it into the data/cifar-x folder
# For cifar-10 the data/cifar-10 folder should contain a data_batch_1, ... and a test_batch file.
# For cifar-100 the data/cifar-100 folder should contain a train and a test file.


def _gen(params, stride=1, offset=0, infinite=False):
    images, labels = params

    loop_condition = True
    while loop_condition:
        for idx in range(offset, len(images), stride):
            img = np.reshape(images[idx], (3, 32, 32))
            yield ({"image": img.transpose((1, 2, 0))}, {"probs": labels[idx]})
        loop_condition = infinite


def cifar(base_dir, phase, version=10):
    images = []
    labels = []

    if version == 10:
        if phase == "train":
            for x in range(1,5,1):
                data_path = os.path.join(base_dir, "data_batch_" + str(x))
                with open(data_path, 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                    images.extend(dict[b"data"])
                    labels.extend(dict[b"labels"])
        if phase == "test":
            data_path = os.path.join(base_dir, "test_batch")
            with open(data_path, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                images.extend(dict[b"data"])
                labels.extend(dict[b"labels"])
    if version == 100:
        data_path = os.path.join(base_dir, phase)
        with open(data_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            images.extend(dict[b"data"])
            labels.extend(dict[b"fine_labels"])

    return _gen, (images, labels)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_data = cifar("data/cifar-10", "train")

    data_fn, data_params = train_data
    data_gen = data_fn(data_params)

    img, label = next(data_gen)
    print("Image shape:")
    print(img["image"].shape)

    for img, label in data_gen:
        plt.imshow(img["image"])
        plt.show()
