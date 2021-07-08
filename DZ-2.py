import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import numpy as np

# импортируем данные и создаем обучающую и тестовую выборки
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

# добавь отображение графика
import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(10):
  plt.subplot(5,2,i+1)
  plt.tight_layout()
  plt.imshow(x_train[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()

num_classes = 10 # цифры 0-9
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) # размер изображения 28x28
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# переводим изображения в диапазон [0,1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# преобразование векторных классов в бинарные матрицы 
y_train = np_utils.to_categorical(y_train, num_classes) # альтернативно keras.utils для других версий Keras
y_test = np_utils.to_categorical(y_test, num_classes)
print('Размер обучающей выборки:', x_train.shape[0])
print('Размер тестовой выборки:', x_test.shape[0])

model = Sequential()
model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(28, 28, 1))) # для других версий Keras может быть в виде 32, (3,3)
# добавь Flatten и два слоя Dense - в первом активационная функция должна быть relu, во втором - softmax
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
# возможно keras.losses.categorical_crossentropy для других версий Keras
print(model.summary())

# добавь fit
# Александр рекомендовал использовать batch_size = 200 и число эпох 25
# раздели train set на непосредственно обучающую и валидационную (20% от размера test set)

#  0.033334% от 60 000 это 2001 пример для валидации, т.е. около 20% от размера тестовой выборки 10 000
epochs=25
batch_size=200
H = model.fit(x_train, y_train,callbacks =[ModelCheckpoint("MNIST.h5",monitor="val_acc",save_best_only=True, save_weights_only=False, mode="auto")], batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.033334, shuffle=True)

# добавь evaluate и напечатай точность работы на тестовых данных с точностью до 2 знаков после запятой
score = model.evaluate(x_test, y_test, verbose=0)
print("Test score: %.2f" % score[0])
print("Test accuracy (score[1): %.2f" % score[1])


plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig('MNIST.png')
plt.show()