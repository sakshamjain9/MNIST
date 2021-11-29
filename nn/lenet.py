from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.models import Sequential


class LeNet:
    def build(self, width, height, depth, classes):
        input_shape = (height, width, depth)
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        model = Sequential()
        model.add(Conv2D(20, (5, 5), input_shape=input_shape, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(50, (5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dense(classes, activation='softmax'))
        return model
