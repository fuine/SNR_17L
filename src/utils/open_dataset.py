import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
# %matplotlib inline

image_ind = 10
train_data = sio.loadmat('../data/train_32x32.mat')

# access to the dict
x_train = train_data['X']
y_train = train_data['y']

# show sample
plt.imshow(x_train[:,:,:,image_ind])
print(y_train[image_ind])
