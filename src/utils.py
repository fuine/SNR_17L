import numpy as np
import scipy.io as sio


def load_dataset(path, gray=False):
    """
    Load dataset from the given path. Images are standardized
    (byte values are divided by 255).

    :param path: path to the dataset
    :param gray: if True it will convert images to grayscale, defaults to False
    :returns: numpy's 4d array with consecutive dimensions being:
              (image_id, im_width, im_height, channel).
    """
    train_data = sio.loadmat(path)
    x_train = train_data['X']
    y_train = train_data['y']%10
    x_train = np.transpose(x_train, (3, 0, 1, 2))
    if not gray:
        x_train = x_train.astype(np.float32)
        x_train /= 255
        return x_train, y_train
    new_x_train = np.empty((x_train.shape[0], 32, 32, 1))
    for i, img in enumerate(x_train):
        new_x_train[i] = rgb2gray(img)
    return new_x_train, y_train


def rgb2gray(rgb):
    """
    Convert image to grayscale and standardize its pixel values.

    :param rgb: numpy array, where the third dimension is color space
    :returns: standardized image in grayscale
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = np.reshape(gray, (rgb.shape[0], rgb.shape[1], 1))
    gray = gray.astype(np.float32)
    gray /= 255
    return gray


def balanced_subsample(y):
    """
    Create random subsample from multiclass dataset with equal number of
    examples in each class. Size of the subsample is equal to the number of
    examples in the smallest class.

    :param y: pandas series of class labels for consecutive examples
    :returns: list of indices of examples to choose
    """
    subsample = []
    n_smp = y.value_counts().min()
    for label in y.value_counts().index:
        samples = y[y == label].index.values
        index_range = range(samples.shape[0])
        indices = np.random.choice(index_range, size=n_smp, replace=False)
        subsample += samples[indices].tolist()
    return subsample


def unison_shuffled_copies(a, b):
    """
    Shuffle two numpy arrays/vectors in unison.

    :param a: first numpy array/vector
    :param b: second numpy array/vector
    :returns: tuple of shuffled arrays
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def select_classes(y, classes):
    """
    Select indices of examples from specified classes.

    :param y: pandas series of class labels for consecutive examples
    :param classes: list of class labels to select
    :returns: list of indices of selected examples
    """
    ret = []
    for label in classes:
        samples = y[y == label].index.values
        ret.extend(samples)
    return ret
