from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def mnist(base_dir, phase, prepare_features=None):
    _mnist = input_data.read_data_sets(base_dir, one_hot=True)
    if phase == "test":
        images = _mnist.test.images
        labels = _mnist.test.labels
    else:
        images = _mnist.train.images
        labels = _mnist.train.labels

    if prepare_features is not None:
        images = prepare_features(images)

    def gen(skip_n=1, offset=0, infinite=False):
        loop_condition = True
        while loop_condition:
            for idx in range(offset, len(images), skip_n):
                yield (images[idx], labels[idx])
            loop_condition = infinite

    return gen


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_data = mnist("data/mnist", "train", lambda x: np.reshape(np.array(x), (-1, 28, 28)))()

    img, label = next(train_data)
    print("Image shape:")
    print(img.shape)

    for img, label in train_data:
        plt.imshow(img)
        plt.show()
