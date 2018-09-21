import numpy as np
from keras import backend as K
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import concatenate
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.models import Model
from keras.utils.layer_utils import convert_all_kernels_in_model
from utils import local_response_normalization
from utils import split_tensor

def AlexNet(weights_path=None, n_classes=200):

    inputs = Input(shape=(3,227,227))
    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
                           name='conv_1')(inputs)

    conv_2 = local_response_normalization(name="LRN1")(conv_1)
    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_2)

    # conv_2 = ZeroPadding2D((2,2))(conv_2)
    a1 = Convolution2D(128,5,5, activation='relu', name='conv_2_1')(split_tensor(ratio_split=2,id_split=0)(conv_2))
    a2 = Convolution2D(128,5,5, activation='relu', name='conv_2_2')(split_tensor(ratio_split=2,id_split=1)(conv_2))
    conv_2 = concatenate([a1,a2], axis=1)
    conv_3 = local_response_normalization(name="LRN2")(conv_2)
    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_3)

    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3')(conv_3)

    conv_3 = ZeroPadding2D((2,2))(conv_3)
    conv_4 = concatenate([Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1))(split_tensor(ratio_split=2,id_split=i)(conv_3)) for i in range(2)], axis=1)

    conv_3 = ZeroPadding2D((2,2))(conv_3)
    conv_5 = concatenate([Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1))(split_tensor(ratio_split=2,id_split=i)(conv_4)) for i in range(2)], axis=1)
    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    if n_classes == 1000:
        dense_3 = Dense(n_classes,name='dense_3')(dense_3)
    else:
        dense_3 = Dense(n_classes,name='dense_3_new')(dense_3)

    prediction = Activation("softmax",name="softmax")(dense_3)

    model = Model(input=inputs, output=prediction)

    if weights_path:
        model.load_weights(weights_path)
    if K.backend() == 'tensorflow':
        convert_all_kernels_in_model(model)

    return model
