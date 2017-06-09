import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda0,floatX=float32"

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import os
import sys
from utils import load_dataset

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

x_train, y_train = load_dataset('../data/train_32x32.mat')
x_test, y_test = load_dataset('../data/test_32x32.mat')

x_train_gray, y_train = load_dataset('../data/train_32x32.mat', gray=True)
x_test_gray, y_test = load_dataset('../data/test_32x32.mat', gray=True)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 1), padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Conv2D(32, (3, 3), activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 1), padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Conv2D(32, (3, 3), activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.8))
model.add(Dense(10, activation='softmax'))

epochs = 100
lrate = 0.04
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

history = model.fit(x_train_gray, y_train, validation_data=(x_test_gray, y_test), epochs=epochs, batch_size=128)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
