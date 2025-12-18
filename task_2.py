import numpy as np
from tensorflow import keras  
from keras import models, layers

# входные данные для нейронки 2 внешних которые содержат 5 чисел
x = np.array([[1, 2, 3, 4, 5],
              [0, 1, 0, 1, 0]])

print("Входы (x):")
print(x) # вывод массива х
print()

# Диапазон для инициализации весов
min_weight = 100
max_weight = 200

# Функции активации для тестирования
activations = ['relu', 'sigmoid', 'tanh']

for activation in activations:
    print(f"Активация: {activation}")

    # Создаём модель
    model = models.Sequential([
        layers.Dense(3, input_shape=(5,), activation=activation)
    ])

    # Получаем текущие веса  случайные или нулевые
    # Нам нужно получить правильные формы
    layer_weights, layer_biases = model.layers[0].get_weights()
    
    # Инициализируем веса в нужном диапазоне
    new_weights = np.random.uniform(low=min_weight, high=max_weight, size=layer_weights.shape).astype(np.float32)
    new_biases = np.zeros_like(layer_biases, dtype=np.float32) 

    # Устанавливаем новые веса
    model.layers[0].set_weights([new_weights, new_biases])

    # Делаем предсказание
    y = model.predict(x, verbose=0) 

    # Выводим результаты
    print(f"Выходы слоя с активацией '{activation}':")
    print(y)
    print()

