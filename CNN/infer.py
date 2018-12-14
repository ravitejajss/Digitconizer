import time
import numpy as np
np.random.seed(123)

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.utils import np_utils

import tensorflow as tf

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('uint8')
X_test = X_test.astype('uint8')

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = tf.keras.models.load_model("my_model.h5")
print(model)
data = X_test[4,:,:,0]
print(Y_test[4,:])
data =data.reshape(1,28,28,1)
start_time = time.time()
out = model.predict(data)
print("--- %s seconds ---" % (time.time() - start_time))
print(out)
