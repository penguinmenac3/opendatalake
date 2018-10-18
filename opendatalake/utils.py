import numpy as np

def one_hot(idx, max_idx):
    label = np.zeros(max_idx, dtype=np.uint8)
    label[idx] = 1
    return label


def crop_center(img, cropy, cropx):
    y, x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx, :]
