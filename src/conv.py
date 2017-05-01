import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import argparse

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
# from nolearn.lasagne.visualize import plot_conv_weights

from utils import load_dataset
from utils import balanced_subsample
from utils import unison_shuffled_copies

np.random.seed(42)

im_size = 32
gray = True


x_train, y_train = load_dataset('../data/train_32x32.mat', gray)
x_test, y_test = load_dataset('../data/test_32x32.mat', gray)
y_train = y_train.flatten()
y_test = y_test.flatten()
y_train[y_train == 10] = 0
y_test[y_test == 10] = 0
x_train = np.transpose(x_train, (0, 3, 1, 2))
x_test = np.transpose(x_test, (0, 3, 1, 2))


# y_d = pd.Series(y_train)
# subinds = select_classes(y_d, [1, 8])
# x_train = x_train[subinds]
# y_train = y_train[subinds]
#
# y_d = pd.Series(y_test)
# subinds = select_classes(y_d, [1, 8])
# x_test = x_test[subinds]
# y_test = y_test[subinds]

print("Bincount in y_train before balancing: {}".format(np.bincount(y_train)))
print("Bincount in y_test before balancing: {}".format(np.bincount(y_test)))
y_d = pd.Series(y_train)
inds = balanced_subsample(y_d)
x_train = x_train[inds]
y_train = y_train[inds]
x_train, y_train = unison_shuffled_copies(x_train, y_train)

# y_train[y_train == 8] = 0
# y_test[y_test == 8] = 0
# print("First 10 labels in y_test: {}".format(y_train[:10]))
# for i in range(10):
#     plt.figure()
#     # plt.imshow(x_train[i])  # , cmap='gray')
#     plt.imshow(np.reshape(x_train[i], (im_size, im_size)), cmap='gray')
# plt.show()

print(x_train[0])
print("x_train shape: {}".format(x_train.shape))
print("x_train dtype: {}".format(x_train.dtype))
print("y_train shape: {}".format(y_train.shape))
print("x_test shape: {}".format(x_test.shape))
print("x_test dtype: {}".format(x_test.dtype))
print("y_test shape: {}".format(y_test.shape))
print("Unique values in y_train: {}".format(np.unique(y_train)))
print("Bincount in y_train: {}".format(np.bincount(y_train)))
print("Bincount in y_test: {}".format(np.bincount(y_test)))
print("First 10 labels in y_test: {}".format(y_test[:10]))

# classic INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC
# note that RELU layers are sewn into conv/dense layers by nonlinearity function
net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('gaussian1', layers.GaussianNoiseLayer),
            ('dense1', layers.DenseLayer),
            ('dropout1', layers.DropoutLayer),
            #  ('dense2', layers.DenseLayer),
            #  ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 1, im_size, im_size),
    # layer conv2d1
    conv2d1_num_filters=32,
    conv2d1_filter_size=(3, 3),
    conv2d1_stride=1,
    conv2d1_pad='same',
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),
    # layer maxpool1
    maxpool1_pool_size=(2, 2),
    # layer conv2d2
    conv2d2_num_filters=64,
    conv2d2_filter_size=(3, 3),
    conv2d2_stride=1,
    conv2d2_pad='same',
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool2_pool_size=(4, 4),
    # gaussian1
    gaussian1_sigma=0.2,
    # dense
    dense1_num_units=2048,
    dense1_nonlinearity=lasagne.nonlinearities.rectify,
    # dropout2
    dropout1_p=0.5,
    #  dense2_num_units=128,
    #  dense2_nonlinearity=lasagne.nonlinearities.rectify,

    #  dropout2_p=0.7,
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,
    # optimization hyperparameters
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=10,
    verbose=1,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', type=str,
                        help='load serialized network from the specified file')
    parser.add_argument(
        '-s', '--save', type=str,
        help='train and save trained network to the specified file')
    parser.add_argument(
        '-v', '--visualize', type=argparse.FileType('rb'),
        help='visualize trained network from the specified file')

    args = parser.parse_args()
    if args.save:
        # Train the network
        nn = net1.fit(x_train, y_train)
        net1.save_params_to(args.save)
    elif args.load:
        # Load the network
        net1.load_params_from(args.load)
        nn = net1
    elif args.visualize:
        pass
        net1.load_params_from(args.load)
        nn = net1
        # plot_conv_weights(nn, figsize=(4, 4))
        exit()
    else:
        nn = net1.fit(x_train, y_train)

    # Classify test set
    preds = nn.predict(x_test)
    print(preds[:10])
    # Print report
    # print(y_test)
    # print(preds)
    print(classification_report(y_test, preds, digits=5))
    cm_ = confusion_matrix(y_test, preds)
    plt.matshow(cm_)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
