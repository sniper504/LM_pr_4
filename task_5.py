import tensorflow as tf
from keras import layers, models, initializers
import matplotlib.pyplot as plt

# Загружаем данные MNIST (цифры от 0 до 9, изображения 28×28)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 #нормализируем данные и делаем значение от 0 до 1
x_train = x_train.reshape(-1, 28*28) # Преобразуем изображения из формата 28×28 в вектор длиной 784
x_test = x_test.reshape(-1, 28*28)

# Функция для создания модели
def make_model(use_bias=True, initializer="glorot_uniform"):
    return models.Sequential([
        layers.Dense(128, activation="relu", use_bias=use_bias, # 128 колво нейронов для обработки
            kernel_initializer=initializer),
        layers.Dense(10, activation="softmax", use_bias=use_bias, # 10 тк 10 цифр
            kernel_initializer=initializer)
    ])

# Список инициализаторов
inits = {
    "GlorotUniform": initializers.GlorotUniform(), #стандартный
    "HeNormal": initializers.HeNormal(), # лучше для ReLU
    "RandomNormal": initializers.RandomNormal(mean=0., stddev=0.05) # случайное норм распределение
}

results = {} # словарь для сохранения истории обучения

for name, init in inits.items(): # цикл по всем инициализаторам
    print(f" Initializer: {name}")
    
    # Модель с bias
    model_bias = make_model(use_bias=True, initializer=init)
    model_bias.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) # spars... функция потерь
    hist_bias = model_bias.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0, # сохраняем историю обучения, verbose управляет тем насколько подробно выводится информация во время обучения
        validation_data=(x_test, y_test))
    
    # Модель без bias
    model_nobias = make_model(use_bias=False, initializer=init)
    model_nobias.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist_nobias = model_nobias.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0,
        validation_data=(x_test, y_test))
    
    # Сохраняем результаты
    results[name] = {
        "bias": hist_bias.history,
        "nobias": hist_nobias.history
    }

    # Выводим распределение весов
    weights_bias = model_bias.layers[0].get_weights()[0].flatten() # достаем веса первого слоя и запихиваем в марицу весов
    weights_nobias = model_nobias.layers[0].get_weights()[0].flatten()

    # bins=50 количество столбцов, alpha=0.6 прозрачность, заголовок какой инициализатор использовался
    plt.figure(figsize=(10,4))
    plt.hist(weights_bias, bins=50, alpha=0.6, label="Bias=True")
    plt.hist(weights_nobias, bins=50, alpha=0.6, label="Bias=False")
    plt.title(f"Распределение весов ({name})")
    plt.legend()
    plt.show()

# Сравнение accuracy/loss
for name in results:
    print(f"Initializer: {name}") # name это строка с названием инициализатора
    print("Bias=True  - Final acc:", results[name]["bias"]["val_accuracy"][-1], # results[name] словарь истории обучения, -1 последнийэлемент списка
          "Loss:", results[name]["bias"]["val_loss"][-1])
    print("Bias=False - Final acc:", results[name]["nobias"]["val_accuracy"][-1], # val_loss список ошибок 
          "Loss:", results[name]["nobias"]["val_loss"][-1])
