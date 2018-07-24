import os
from scipy.misc import imread
import matplotlib.pyplot as plt
try:
    from IPython.display import clear_output
    NO_IPYTHON = False
except ModuleNotFoundError:
    NO_IPYTHON = True

PHASE_TRAIN = "train"
PHASE_VALIDATION = "validation"


def _gen(params, stride=1, offset=0, infinite=False):
    file_objs, phase, base_dir, return_paths = params

    loop_condition = True
    while loop_condition:
        for idx in range(offset, len(file_objs), stride):
            filename = file_objs[idx]["filename"]
            drive = file_objs[idx]["drive"]
            image = os.path.join(base_dir, phase, drive, "proj_depth", "groundtruth", "image_02", filename)
            depth_image = os.path.join(base_dir, phase, drive, "proj_depth", "groundtruth", "image_02", filename)

            feature = imread(image, mode="RGB")
            label = imread(depth_image) / 100.0
            label[label>100] = 100
            if return_paths:
                yield ({"image": feature, "imagepath": image},
                       {"depth": label, "depthpath": depth_image})
            else:
                yield ({"image": feature},
                       {"depth": label})
        loop_condition = infinite


def kitti_depth(base_dir, phase, return_paths=False):
    if phase == PHASE_VALIDATION:
        phase = "val"

    drives = [f for f in os.listdir(os.path.join(base_dir, phase))]
    file_objs =[]
    for drive in drives:
        filenames = [f for f in os.listdir(os.path.join(base_dir, phase, drive, "proj_depth", "groundtruth", "image_02")) if f.endswith(".png")]
        for f in filenames:
            file_objs.append({"filename": f, "drive": drive})

    return _gen, (file_objs, phase, base_dir, return_paths)

if __name__ == "__main__":
    print("Loading Dataset:")
    train_data = kitti_depth("datasets/kitti_depth", phase=PHASE_TRAIN)

    data_fn, data_params = train_data
    data_gen = data_fn(data_params)

    for feat, label in data_gen:
        fig = plt.figure(figsize=(24, 14), dpi=80)
        fig.add_subplot(2, 1, 1)
        plt.title("Image")
        plt.imshow(feat["image"])
        fig.add_subplot(2, 1, 2)
        plt.title("Depth")
        plt.imshow(label["depth"], cmap="jet")
        #plt.colorbar()
        plt.show()
        plt.close(fig)
