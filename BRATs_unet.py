# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 07:49:08 2017

@author: THANHHAI
"""
# Same as the original U-Net model with input (240, 240, 1)

from __future__ import print_function
import os
# force tensorflow to use CPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from skimage import data, util
# from skimage.measure import label, regionprops
from skimage import io
# from skimage.transform import resize
import SimpleITK as sitk
from matplotlib import pyplot as plt
# import subprocess
import random
# import progressbar
from glob import glob
import gc


import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dropout, Activation, Reshape
from keras.layers import Input, Conv2DTranspose
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
# from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from BRATs_data_unet import load_train_data

# TF dimension ordering in this code
# K.set_image_data_format('channels_last')  

# an argument to your Session's config arguments, it might help a little, 
# though it won't release memory, it just allows growth at the cost of some memory efficiency
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
# with tf.Session(config = config) as s:
sess = tf.Session(config=config)

# -------------------------------------------------------------
smooth = 1.
nclasses = 5 # no of classes, if the output layer is softmax
# nclasses = 1 # if the output layer is sigmoid
img_rows = 240
img_cols = 240

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true.astype('float32'))
    y_pred_f = K.flatten(y_pred.astype('float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def cnnBRATsInit_unet():  
    # the output layer is softmax, loss should be categorical_crossentropy
    # Downsampling 2^4 = 16 ==> 240/16 = 15
            
    inputs = Input((img_rows, img_cols, 1))
    # model.add(Dense((256, 256), input_shape=(240, 240)))
    # Block 1
    conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(inputs)
    # conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Block 2
    conv2 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(pool1)
    # conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # this MaxPooling2D has the same result with the above line.
    # pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2)
    
    # W_new = (W - F + 2P)/S + 1
    # W_new = (120 - 3 + 2*1)/2 + 1 = 60.5 --> 60 (round down)
    # P = 1, because the padding parameter is 'same', the size of W has no change
    # before divide by Stride parameter
    
    
    # Block 3
    conv3 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', 
                   kernel_regularizer=l2(0.01))(pool2)
    # conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', 
                   kernel_regularizer=l2(0.01))(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # dropout1 = Dropout(0.5)(pool3)
    
    # Block 4
    conv4 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', 
                   kernel_regularizer=l2(0.01))(pool3)
    # conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', 
                   kernel_regularizer=l2(0.01))(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # dropout2 = Dropout(0.5)(pool4)
    
    # Block 5
    conv5 = Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(pool4)
    # conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv5)
    conv5 = BatchNormalization()(conv5)
    # pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(conv5)
    # dropout3 = Dropout(0.5)(pool5)
    
    # Block 6
    # axis = -1 same as axis = last dim order, in this case axis = 3
    up6 = concatenate([Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(up6) 
    # conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv6)
    conv6 = BatchNormalization()(conv6)
    # dropout4 = Dropout(0.5)(conv6)
    
    # Block 7
    up7 = concatenate([Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(up7) 
    # conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv7)
    conv7 = BatchNormalization()(conv7)
    
    # Block 8
    up8 = concatenate([Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(up8)
    # conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv8)
    conv8 = BatchNormalization()(conv8)
    
    # Block 9
    up9 = concatenate([Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(up9) 
    # conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv9)
    conv9 = BatchNormalization()(conv9)
    
    # Block 10
    conv10 = Conv2D(nclasses, kernel_size=(1, 1), padding='same')(conv9)
    
    # Block 10
    # curr_channels = nclasses
    _, curr_width, curr_height, curr_channels = conv10._keras_shape
    out_conv10 = Reshape((curr_width * curr_height, curr_channels))(conv10)
    act_soft = Activation('softmax')(out_conv10)
    
    model = Model(inputs=[inputs], outputs=[act_soft])


    sgd = SGD(lr=0.0001, decay=0.01, momentum=0.9, nesterov=True)
    # sgd = SGD(lr=0.0001, decay=1, momentum=0.9, nesterov=True)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) 
    
    # model.compile(optimizer='sgd', loss=dice_coef_loss, metrics=[dice_coef])
    
    # model.summary()
    return model
    
def preprocessing(imgs):
    # insz_h, insz_w = imgs.shape[1], imgs.shape[2] # row, col
    # imgs = imgs.reshape(imgs.shape[0], insz_h, insz_w, 1)   
    imgs = imgs[..., np.newaxis] # result is the same
    return imgs

def save_trained_model(model):    
    # serialize model to JSON
    model_json = model.to_json()
    with open("cnn_BRATs_unet.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("cnn_BRATs_unet.h5")
    print("Saved model to disk")
    
def show_img(imgs_bef, imgs_aft):
    # Show image after augmenting data
    for n in range(imgs_bef.shape[0]):
        fig, axes = plt.subplots(ncols=2)
        ax = axes.ravel()
        ax[0].imshow(imgs_bef[n], cmap=plt.cm.gray)
        ax[1].imshow(imgs_aft[n], cmap=plt.cm.gray)
        plt.show()
    
def augmenting_data(in_data):
    # ZCA whitening
    # in_data should be [samples][width][height][channels]
    
    # convert to float
    # in_data = in_data.astype('float32')
    out_data = in_data
    
    # define data preparation
    datagen = ImageDataGenerator(zca_whitening=True)
    # fit parameters from data
    datagen.fit(out_data)
    
    # show result    
    show_img(in_data, out_data)
    
    return out_data
    
    
def train_network():
    print('Loading and preprocessing train data...')
    imgs_train, imgs_label_train = load_train_data('softmax')
          
    imgs_train = preprocessing(imgs_train)
    
    # print('Augmenting train data...')
    # imgs_train_aug = augmenting_data(imgs_train)
    print('Calculating mean and std of train data...')
    imgs_train = imgs_train.astype('float32') 
    imgs_train /= 255. 
    
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    # print('mean = %f, std = %f' %(mean, std))    
    # np.save('imgs_train_mean_std.npy', [mean, std])
    
    imgs_train -= mean
    imgs_train /= std    
    
    minv = np.min(imgs_train)  # mean for data centering
    maxv = np.max(imgs_train)  # std for data normalization
    print('min = %f, max = %f' %(minv, maxv)) 
    
    # mean2, std2 = np.load('imgs_train_mean_std.npy')
    # print('mean = %f, std = %f' %(mean2, std2))
    
    # define data preparation
    # datagen = ImageDataGenerator(zca_whitening=True)
    batch_size = 2
    # datagen = ImageDataGenerator(rescale=1./255, rotation_range=40,
                                 # width_shift_range=0.2, height_shift_range=0.2,
                                 # shear_range=0.2, fill_mode='nearest')
    # fit parameters from data
    # datagen.fit(imgs_train)     
    # train_generator = datagen.flow(imgs_train, imgs_label_train, batch_size=batch_size)
    
    print('Imgs train shape', imgs_train.shape)
    
    # convert classes from (0-4) to uint8, if activation is softmax
    # no need to preprocess for label data
    # imgs_label_train = imgs_label_train.astype('uint8')
    # imgs_label_train = preprocessing(imgs_label_train, True)
    print('Imgs label shape', imgs_label_train.shape)
    
    # scale classes from (0-4) to (0-1), if activation is sigmoid
    # imgs_label_train = imgs_label_train.astype('float32')
    # imgs_label_train /= 4. 
    # gmax = np.max(imgs_label_train)
    # print(gmax)
    
    # print('Loading and preprocessing test data...')
    # imgs_test, imgs_label_test = load_test_data('softmax')
    # imgs_test = preprocessing(imgs_test)
    # imgs_test = imgs_test.astype('float32') 
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    # test_datagen = ImageDataGenerator(rescale=1./255)
    # fit parameters from data
    # test_datagen.fit(imgs_test)     
    # val_generator = test_datagen.flow(imgs_test, imgs_label_test, batch_size=batch_size)
    
    print('Creating and compiling model...')
    # model = cnnBRATsInit_holes()
    model = cnnBRATsInit_unet()
    model.summary()
    
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    
    print('Fitting model...')
    model.fit(imgs_train, imgs_label_train, batch_size=batch_size, epochs=20, verbose=1, shuffle=True,
              validation_split=0.05,
              callbacks=[model_checkpoint])
    
    #model.fit_generator(train_generator,
                        # validation_data=val_generator,
                        # validation_steps=imgs_test.shape[0] // batch_size,
                        # steps_per_epoch=imgs_train.shape[0] // batch_size,
                        # samples_per_epoch=imgs_train.shape[0],
                        # nb_epoch=20,
                        # verbose=1,
                        # callbacks=[model_checkpoint])
    
    save_trained_model(model)
    
    # useless
    # del history
    # del model
    # gc.collect()
    
    # print('Evaluating model...')
    # scores = model.evaluate(imgs_train, imgs_label_train, batch_size=4, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
if __name__ == '__main__':
    train_network()
