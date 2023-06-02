# -*- coding: utf-8 -*-
"""temperatureC2F.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1x8y2FFrl25ZSc2ruMt1cXXTNCeNEBzr_
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Datos de entrada
c = np.array([0.4, 0.44, 1.02, 1.13, 2.13, 3.59, 4.01, 4.67, 5.46, 5.63])
# Datos de Salida
f = np.array([31.7, 33.89, 32.8, 35.1, 34.8, 39.5, 38.2, 41.5, 40.8, 43.2])

## Configuración de una Capa
# units: numero de neuronas de la capa
# input_shape: numero de entradas
layers = [
    tf.keras.layers.Dense(units=8, input_shape=[1]),
    tf.keras.layers.Dense(units=8),
    tf.keras.layers.Dense(units=8),
    tf.keras.layers.Dense(units=1),
]

## Diseño del Modelo
model = tf.keras.Sequential(layers)

## Compilación del Modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error"
)

# Entrenamiento del modelo
training = model.fit(c, f, epochs=10000, verbose=False)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(training.history["loss"])

result = model.predict([100.0])
print(result)