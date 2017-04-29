import numpy as np
import scipy.io as sio


def load_dataset(path, gray=False):
    train_data = sio.loadmat(path)
    x_train = train_data['X']
    y_train = train_data['y']
    x_train = np.transpose(x_train, (3, 0, 1, 2))
    if not gray:
        return x_train, y_train
    new_x_train = np.empty((x_train.shape[0], 32, 32))
    for i, img in enumerate(x_train):
        new_x_train[i] = rgb2gray(img)
    return new_x_train, y_train


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
