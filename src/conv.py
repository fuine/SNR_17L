from keras.models import Sequential
from keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    GaussianNoise,
    Dropout,
    Flatten
)

from keras.optimizers import SGD
from keras.models import load_model
from keras.callbacks import (
    ReduceLROnPlateau,
    ModelCheckpoint
)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import argparse
import itertools
import os

from utils import load_dataset
# from utils import balanced_subsample
from utils import unison_shuffled_copies

from keras.utils import plot_model
from resnet import ResnetBuilder

# fix random seed for reproducibility
np.random.seed(42)
im_size = 32
gray = True
resnet = True
verbosity = 2
plot = True
# To create validation set from train set set this parameter to the percentage
# of the validation set (eg 0.2 will create a validation set from 20% of the
# training set)
valid_size = 0.1

if gray:
    channel_nums = 1
else:
    channel_nums = 3


def probas_to_classes(proba):
    """
    Changes matrix of probabilities to array of predicted classes.
    """
    if proba.shape[-1] > 1:
        return proba.argmax(axis=-1)
    else:
        return (proba > 0.5).astype('int32')


def plot_confusion_matrix(
        cm, classes, normalize=False, title='Confusion matrix',
        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_history(history):
    """
    Plots history from the given History object.
    """
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.figure()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')


def load_data():
    x_train, y_train = load_dataset('../data/train_32x32.mat', gray)
    x_test, y_test = load_dataset('../data/test_32x32.mat', gray)
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0

    # Shuffle train examples
    x_train, y_train = unison_shuffled_copies(x_train, y_train)

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
    y_train = np.reshape(y_train, (-1, 1))
    y_test = np.reshape(y_test, (-1, 1))

    return x_train, y_train, x_test, y_test


def get_model(checkpoint_path=None):
    # create model
    if resnet:
        model = ResnetBuilder.build_resnet_18((channel_nums, im_size, im_size),
                                              10)
    else:
        model = Sequential([
            Conv2D(filters=32, kernel_size=3, strides=1, padding="same",
                   kernel_initializer='glorot_normal', activation='relu',
                   input_shape=(im_size, im_size, channel_nums)),
            MaxPooling2D(pool_size=2),
            Conv2D(filters=64, kernel_size=3, strides=1, padding="same",
                   activation='relu'),
            MaxPooling2D(pool_size=4),
            GaussianNoise(stddev=0.2),
            Flatten(),
            Dropout(rate=0.5),
            Dense(units=2048, activation='relu'),
            Dropout(rate=0.5),
            Dense(units=10, activation='softmax')
        ])

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', type=str,
                        help='load serialized network from the specified file')
    parser.add_argument(
        '-s', '--save', type=str,
        help='train and save trained network to the specified file')
    parser.add_argument(
        '-v', '--visualize', type=str,
        help='visualize trained network from the specified file')
    parser.add_argument(
        '-i', '--inc_train', type=str,
        help='incrementally train the network with n epochs,'
        ' requires -l argument, saves uptrained network to the given file'
    )
    parser.add_argument(
        '-e', '--epochs', type=int,
        default=10,
        help='number of epochs, defaults to 10'
    )

    args = parser.parse_args()

    x_train, y_train, x_test, y_test = load_data()
    class_weight = class_weight.compute_class_weight(
        'balanced', np.unique(y_train.flatten()), y_train.flatten())
    print(class_weight)

    if valid_size is not None:
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=valid_size,
            random_state=42, stratify=y_train)
        validation_data = (x_valid, y_valid)
    else:
        validation_data = None

    callbacks = []
    reduce_lr = ReduceLROnPlateau(
        verbose=1, patience=3, min_lr=0.001)
    callbacks.append(reduce_lr)

    # TODO check if paths are valid before calculations
    if args.save:
        # Train the network
        filename, file_extension = os.path.splitext(args.save)
        model = get_model()
        checkpoint = "{}_chk_{{epoch:02d}}-{{val_loss:.2f}}.{}".format(
            filename, file_extension)
        callbacks.append(ModelCheckpoint(checkpoint, verbose=1,
                                         save_best_only=True, period=3))
        print(model.summary())
        history = model.fit(x_train, y_train, epochs=args.epochs,
                            class_weight=class_weight,
                            validation_data=validation_data,
                            callbacks=callbacks,
                            verbose=verbosity)
        model.save(args.save)
        if plot:
            plot_history(history)
    elif args.inc_train:
        if args.load is None:
            to_load = args.inc_train
        else:
            to_load = args.load
        model = load_model(to_load)
        filename, file_extension = os.path.splitext(args.inc_train)
        checkpoint = "{}_chk_{{epoch:02d}}-{{val_loss:.2f}}{}".format(
            filename, file_extension)
        callbacks.append(ModelCheckpoint(checkpoint, verbose=1,
                                         save_best_only=True, period=3))
        print(model.summary())
        history = model.fit(x_train, y_train, epochs=args.epochs,
                            class_weight=class_weight,
                            validation_data=validation_data,
                            callbacks=callbacks,
                            verbose=verbosity)
        model.save(args.inc_train)
        if plot:
            plot_history(history)
    elif args.load:
        # Load the network
        model = load_model(args.load)
        print(model.summary())
    elif args.visualize:
        model = load_model(args.visualize)
        plot_model(model, to_file='model.png', show_shapes=True)
        exit()
    else:
        model = get_model()
        print(model.summary())
        history = model.fit(x_train, y_train, epochs=args.epochs,
                            class_weight=class_weight,
                            validation_data=validation_data,
                            callbacks=callbacks,
                            verbose=verbosity)
        if plot:
            plot_history(history)

    probas = model.predict(x_test, verbose=verbosity)
    preds = probas_to_classes(probas)
    print(classification_report(y_test, preds, digits=5))
    cm_ = confusion_matrix(y_test, preds)
    plot_confusion_matrix(cm_, classes=np.unique(y_test).tolist(),
                          title='Confusion matrix, without normalization')
    plt.show()
