from keras import models, layers
import numpy as np

models = models.Sequential([
    layers.Dense (3, input_shape = (5,)), activation = 'relu'
])

weights = np.full((5,3),0.5)
biases =np.zeros(3)
models.layers[0].set_weights([weights,biases])
x = np.array([1,2,3,4,5], [0,1,0,1,0])
y = models.predict(x)
print("Enters:", x)
print("Outputs:", y)
weights, biases = models.layers[0].get_weights()

print(biases.shape)
print(weights.shape)
print(weights[:3,:5])# матрица весов
print(biases[:10])# первые 10 сдвигов

