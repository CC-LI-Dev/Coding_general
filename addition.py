#
import tensorflow as tf
from keras.api._v2 import keras
import random

# data

x_train = []
y_train = []
for i in range(1000):
    x_0 = random.randint(0, 100)
    x_1 = random.randint(0, 100)
    y = x_0 + x_1

    x_train.append((x_0, x_1))
    y_train.append(y)

# model
inputs = keras.Input(2)
dense_layer = keras.layers.Dense(1, use_bias=False)
outputs = dense_layer(inputs)

model = keras.Model(inputs, outputs)
model.summary()

print(f"weights before training{dense_layer.get_weights()}")
# training

model.compile(loss="mse")
model.fit(x_train, y_train, epochs=10000)
model.save("model_addition")

print(f"weights after training{dense_layer.get_weights()}")
