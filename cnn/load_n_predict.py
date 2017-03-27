# How to load and use weights from a checkpoint
import my_preprocessing
import my_keras_models
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy

# import data
out_x, out_y = my_preprocessing.preprocessing(clean=True,resize=False)


from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import pandas as pd
import cv2
import my_keras_models
from keras.utils import np_utils
from keras import backend as K


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

# data with shape (examples, 1d array)
def resizeData(data,fx=(0.5),fy=(0.5)):
    resized = []
    for i in range(len(data)):
        resized.append(cv2.resize(data[i]),None,fx=fx,fy=fy,interpolation=cv2.INTER_AREA)
    resized = np.array(resized, dtype='float32')    
    return resized


# put all the preprocessing into its own function, makes things a lot easier
def preprocessing():
    # import images
    X = np.fromfile('../test_x.bin', dtype='uint8')
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
    out_x = resized.reshape((100000,784))
    out_y = y
    print ('Resizing done')
    print ('Preprocessing Done')
    return out_x

# the data, shuffled and split between train and test sets
X_kaggle = out_x

batch_size = 200
nb_classes = 19
nb_epoch = 100

#input image dimensions
img_rows, img_cols = 28,28

# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size regular
kernel_size = (5, 5)
# convolution kernel size small
kernel_size_small = (3, 3)

if K.image_dim_ordering() == 'th':
    X_kaggle = X_kaggle.reshape(X_kaggle.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_kaggle = X_kaggle.reshape(X_kaggle.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_kaggle = X_kaggle.astype('float32')
X_kaggle /= 255
print('X_kaggle shape:', X_kaggle.shape)
print(X_kaggle.shape[0], 'kaggle samples')

# load cnn
cnn = my_keras_models()
model = cnn.model_5_2()

# load weights
model.load_weights("./t5_2_mdlckpt/t5_2-20-0.97.hdf5")
# Compile model (required to make predictions)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Created model and loaded weights from file")

# make predicitons
predictions = model.predict(X_kaggle, batch_size=32, verbose=0)


# estimate accuracy on whole dataset using loaded weights
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))