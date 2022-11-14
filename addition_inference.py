from keras.api._v2 import keras

model = keras.models.load_model("model_addition")

print(f"3+7={model.predict([(3,7)])}")
