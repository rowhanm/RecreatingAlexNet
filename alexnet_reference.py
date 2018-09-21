import numpy as np
from keras import backend as K
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils.layer_utils import convert_all_kernels_in_model
from utils import local_response_normalization

def CaffeNet(weights_path=None, n_classes=200):
    model = Sequential()
    model.add(Conv2D(filters=96, input_shape=(3,227,227), kernel_size=(11,11), strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    model.add(local_response_normalization(name="LRN1"))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

    model.add(Conv2D(filters=256, kernel_size=(5,5), padding='valid'))
    model.add(Activation('relu'))
    model.add(local_response_normalization(name="LRN2"))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(filters=384, kernel_size=(3,3), padding='valid'))
    model.add(Activation('relu'))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(filters=384, kernel_size=(3,3), padding='valid'))
    model.add(Activation('relu'))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

    model.add(Flatten())

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    if n_classes != 1000:
        model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    # model.fit(x, y, batch_size=64, epochs=1, verbose=1, validation_split=0.2, shuffle=True)

    if weights_path:
        model.load_weights(weights_path)
    if K.backend() == 'tensorflow':
        convert_all_kernels_in_model(model)

    return model
