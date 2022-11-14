import numpy as np
from keras.api._v2 import keras

# data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = np.expand_dims(x_train, axis=-1) / 255.
x_test = np.expand_dims(x_test, axis=-1) / 255.
# model
inputs = keras.Input((28, 28, 1))  # 1 ist hier Farbkanal # 28, 28 ist x& y bei z käme vor 1
conv_layer = keras.layers.Conv2D(8, 3)
conv = conv_layer(inputs)
pool_layer = keras.layers.MaxPooling2D()
pool = pool_layer(conv)
features = keras.layers.Flatten()(pool)
dense_layer = keras.layers.Dense(10, activation="softmax")
output = dense_layer(features)

model = keras.Model(inputs, output)
model.summary()

# training
model.compile(loss="categorical_crossentropy", metrics="accuracy")
model.fit(x_train, keras.utils.to_categorical(y_train), validation_data=(x_test, keras.utils.to_categorical(y_test)), epochs=10)
