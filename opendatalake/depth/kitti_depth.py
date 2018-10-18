import os
from scipy.misc import imread

from opendatalake.simple_sequence import SimpleSequence

PHASE_TRAIN = "train"
PHASE_VALIDATION = "validation"


class KittiDepth(SimpleSequence):
    def __init__(self, hyperparams, phase, preprocess_fn=None, augmentation_fn=None):
        super(KittiDepth, self).__init__(hyperparams, phase, preprocess_fn, augmentation_fn)
        if phase == PHASE_VALIDATION:
            phase = "val"
        self.phase = phase
        base_dir = self.hyperparams.problem.data_path

        drives = [f for f in os.listdir(os.path.join(base_dir, phase))]
        file_objs =[]
        for drive in drives:
            filenames = [f for f in os.listdir(os.path.join(base_dir, phase, drive, "proj_depth", "groundtruth", "image_02")) if f.endswith(".png")]
            for f in filenames:
                file_objs.append({"filename": f, "drive": drive})

        self.file_objs = file_objs
        self.base_dir = base_dir

    def __num_samples(self):
        return len(self.file_objs)

    def __get_sample(self, idx):
        filename = self.file_objs[idx]["filename"]
        drive = self.file_objs[idx]["drive"]
        image = os.path.join(self.base_dir, self.phase, drive, "proj_depth", "groundtruth", "image_02", filename)
        depth_image = os.path.join(self.base_dir, self.phase, drive, "proj_depth", "groundtruth", "image_02", filename)

        feature = imread(image, mode="RGB")
        label = imread(depth_image) / 100.0
        label[label>100] = 100
        return ({"image": feature, "imagepath": image},
                {"depth": label, "depthpath": depth_image})
