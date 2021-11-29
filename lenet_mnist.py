from nn import LeNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


print('[INFO] Loading Data')
df = pd.read_csv('data/train.csv')
labels = np.array(df['label'])

if K.image_data_format() == 'channels_last':
    data = np.array(df.iloc[:, 1:]).reshape((-1, 28, 28, 1)).astype('float') / 255
else:
    data = np.array(df.iloc[:, 1:]).reshape((-1, 1, 28, 28)).astype('float') / 255

train_X, test_X, train_Y, test_Y = train_test_split(data, labels, test_size=0.25, random_state=42)

lb = LabelBinarizer()
train_Y = lb.fit_transform(train_Y)
test_Y = lb.transform(test_Y)

print('[INFO] Compiling LeNet')
model = LeNet.build(28, 28, 1, 10)
sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

print('[INFO] Training LeNet')
H = model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=100, batch_size=128, verbose=1)

print('[INFO] Evaluating LeNet')
predictions = model.predict(test_X, batch_size=128)
print(classification_report(test_Y.arg_max(axis=1),
      predictions.argmax(axis=1),
      target_names=[str(x) for x in le.classes_]))

print('[INFO] Plotting Loss/Accuracy')
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(100), H.history['loss'], label='train_loss')
plt.plot(np.arange(100), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(100), H.history['accuracy'], label='train_accuracy')
plt.plot(np.arange(100), H.history['val_accuracy'], label='val_accuracy')
plt.title('Lenet Loss/Accuracy')
plt.xlabel('#Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()
