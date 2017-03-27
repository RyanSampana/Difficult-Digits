from __future__ import print_function
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf
tf.python.control_flow_ops = tf

# function to clean an image
def cleanImage(image):
    mean = cv2.meanStdDev(image)[0]
    std = cv2.meanStdDev(image)[1]
    if (mean > 220):
        ret,image = cv2.threshold(image,min(mean+std+5,250),255,cv2.THRESH_BINARY)
    else:
        ret,image = cv2.threshold(image,min(max(mean+2*std,150),245),255,cv2.THRESH_BINARY)
    return image

# function to clean the data
def cleanData(data):
    clean_data = []
    for i in range(len(data)):
        clean_data.append(cleanImage(data[i]))
    clean_data = np.array(clean_data, dtype='float32')
    clean_data = clean_data.reshape((100000,3600))
    return clean_data

# import labels
df = pd.read_csv('../train_y.csv', dtype='int32')
y = np.array(df['Prediction'])

# import images
X = np.fromfile('../train_x.bin', dtype='uint8')
X = X.reshape((100000,60,60))

# clean the data
print ('Cleaning Data')
clean_data = cleanData(X)
print ('Cleaning Done')

# resize the data so we do calculations faster ==> moar experiments
print ('Resizing')
resized = []
for i in range(len(clean_data)):
    resized.append(cv2.resize(clean_data[i].reshape((60,60)),None,fx=(28./60.),fy=(28./60.),interpolation=cv2.INTER_AREA))
resized = np.array(resized)
resized = resized.reshape((100000,784))
print ('Resizing done')

# uncomment the line below to show an example image
# plt.imshow(clean_data[0].reshape((60,60)),cmap=plt.cm.gray_r)

# the data, shuffled and split between train and test sets
max_examples = 70000
X_train = resized[:max_examples]
y_train = y[:max_examples]
X_test = resized[max_examples:]
y_test = y[max_examples:]

batch_size = 128
nb_classes = 19
nb_epoch = 40

#input image dimensions
img_rows, img_cols = 28,28
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (5, 5)

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

# CNN
model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))

model.add(Convolution2D(32, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))


model.add(Flatten())
# dense layer
model.add(Dense(512))
model.add(Activation('relu'))

# drop out layer
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# logging and early stopping
csv_logger = callbacks.CSVLogger('training_3_deep_55.log')
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=2)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test), callbacks=[csv_logger,early_stopping])
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
