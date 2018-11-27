# Digitconizer

Android app that detects handwritten digits.

The app uses a  sequential TensorFlow model and the structure of the network is as follows.
1. Convolution2D layer with activation 'relu' with 3 filters of size 3 and input_shape 28x28x1 output shape 32
2. Convolution2D layer with activation 'relu' and 3 filters of size 3
3. MaxPooling2D layer of pool_size (2,2)
4. Dropout layer of size 0.25
5. Flatten layer
6. Dense layer of size 128 with activation 'relu'
7. Dropout layer of size 0.5
8. Output Dense layer of size 10 with activation 'softmax'

The app uses TensorFlow Lite Interpreter in Android Studio. The model is Interpreted from the tflite file for on device inferencing.