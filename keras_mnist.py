# coding=utf-8
'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import logging
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger('missinglink').addHandler(logging.StreamHandler())

import numpy as np
import missinglink
np.random.seed(1337)  # for reproducibility
import argparse
import os
import pwd


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import Callback
from keras.utils import np_utils
from keras import backend as K
from random import randint

class TestCallback(Callback):
    def __init__(self, test_data, callback):
        self.test_data = test_data
        self.callback = callback

    def on_epoch_end(self, epoch, logs={}):
        X_test, Y_test = self.test_data
        with self.callback.test(self.model):
            score = model.evaluate(X_test, Y_test, verbose=0)



parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--owner-id', required=True)
parser.add_argument('--project-token', required=True)
parser.add_argument('--epochs', type=int, default=8)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--is-sampling', type=bool, default=False)
parser.add_argument('--host')

args = parser.parse_args()

batch_size = args.batch_size
nb_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print('args.is_sampling: ', args.is_sampling, ' type: ', type(args.is_sampling))

if args.is_sampling:
    random_sampling_factor = randint(0, 100) / float(100)

    rows_to_delete = range(int(random_sampling_factor * X_train.shape[0]))

    X_train = np.delete(X_train, rows_to_delete, 0)
    y_train = np.delete(y_train, rows_to_delete, 0)

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(
    nb_filters, kernel_size[0], kernel_size[1],
    input_shape=input_shape))

model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adadelta',
    metrics=['accuracy', 'categorical_accuracy', 'mean_squared_error', 'hinge', 'precision', 'fbeta_score'])

callback = missinglink.KerasCallback(owner_id=args.owner_id, project_token=args.project_token, host=args.host)

callback.set_properties(display_name='KerasMinstTest', description='cool kerassing around')

if args.is_sampling:
    callback.set_hyperparams(sampling_factor=1 - random_sampling_factor) # we log how many samples in % we have from the total samples

callback.set_hyperparams(train_sample_count=X_train.shape[0])
callback.set_hyperparams(test_sample_count=X_test.shape[0])
callback.set_hyperparams(total_epochs=args.epochs)

model.fit(
    X_train, Y_train, batch_size=batch_size, nb_epoch=args.epochs, validation_split=0.2,
    callbacks=[callback, TestCallback((X_test, Y_test), callback)])

with callback.test(model):
    score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
