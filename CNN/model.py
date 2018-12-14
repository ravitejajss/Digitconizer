import numpy as np
np.random.seed(123)
from matplotlib import pyplot as plt
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('uint8')
X_test = X_test.astype('uint8')

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28,1), data_format="channels_last"))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
	      optimizer='adam',
	      metrics=['accuracy'])

model.fit(X_train, Y_train,
	epochs=10, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=1)
print(score)

model.save("my_model.h5")

data = X_test[4,:,:,0]
print(Y_test[4,:])
data =data.reshape(1,28,28,1)
start_time = time.time()
out = model.predict(data)
print("--- %s seconds ---" % (time.time() - start_time))
print(out)


data = X_test[0,:,:,0]
print(Y_test[4,:])
data =data.reshape(1,28,28,1)
start_time = time.time()
out = model.predict(data)
print("--- %s seconds ---" % (time.time() - start_time))
print(out)


data = X_test[1,:,:,0]
print(Y_test[4,:])
data =data.reshape(1,28,28,1)
start_time = time.time()
out = model.predict(data)
print("--- %s seconds ---" % (time.time() - start_time))
print(out)


data = X_test[2,:,:,0]
print(Y_test[4,:])
data =data.reshape(1,28,28,1)
start_time = time.time()
out = model.predict(data)
print("--- %s seconds ---" % (time.time() - start_time))
print(out)
