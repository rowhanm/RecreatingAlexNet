from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from utils import preprocess_image_batch, load_images
import numpy as np
import os

from alexnet_og import AlexNet
from alexnet_reference import CaffeNet

num_classes = 200
path = '<path_to_tiny-imagenet-200>'
X_train, y_train, X_test, y_test = load_images(path, num_classes)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

num_samples=len(X_train)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

model = CaffeNet(n_classes=200)
sgd = SGD(lr=0.01, decay=0.0005, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd,  metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_test, Y_test),  epochs=90, batch_size=32)
