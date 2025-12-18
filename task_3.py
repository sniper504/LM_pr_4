from tensorflow import keras

input_shape = (10,)
units = 16 

# создаем модель из последовательных слоев
model = keras.Sequential([
    keras.layers.Dense(units=units, input_shape=input_shape) 
])

layer = model.layers[0]# первый слой
weights, biases = layer.get_weights() # матрица весов и вектор смещений

# Показываем размеры матрицы весов и вектора смещений.
print("Input shape:", input_shape)
print("Units:", units) # количество нейронов
print("Weights shape:", weights.shape)
print("Bias shape:", biases.shape)


