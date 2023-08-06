import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.utils as tku
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print(mnist)
print(mnist['data'][0])
X , Y = mnist['data'], mnist['target']
print(X)
print(Y)
X.shape
Y.shape
demo_digit = X[675]
demo_digit = demo_digit.reshape(28,28)
plt.imshow(demo_digit, cmap = matplotlib.cm.binary, interpolation='nearest')
print(Y[675])
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 0)
cnn = tf.keras.models.Sequential()
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Dense(64, activation = 'relu', input_dim = 784))
cnn.add(tf.keras.layers.Dense(64, activation = 'relu'))
cnn.add(tf.keras.layers.Dense(10, activation = 'softmax'))
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
cnn.fit(X_train, tku.to_categorical(Y_train), epochs = 5, batch_size = 32)
cnn.evaluate(X_test, tku.to_categorical(Y_test))
test_image = X[675]
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
print(result)
