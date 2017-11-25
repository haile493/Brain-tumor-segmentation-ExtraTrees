# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:11:51 2017

@author: THANHHAI
"""
# Same as the original U-Net model with input (240, 240, 1)

from __future__ import print_function
import numpy as np
from skimage import data, util
from skimage.measure import label, regionprops
from skimage import io
from skimage.transform import resize
from skimage.filters import sobel
import SimpleITK as sitk
from matplotlib import pyplot as plt
# import subprocess
# import random
# import progressbar
from glob import glob
import os
import re

# import tensorflow as tf
# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten, Merge, Reshape, MaxoutDense
# from keras.layers import Conv2D, MaxPooling2D
# from keras.optimizers import SGD
from keras.utils import np_utils, to_categorical
# from keras import backend as K

# an argument to your Session's config arguments, it might help a little, 
# though it won't release memory, it just allows growth at the cost of some memory efficiency
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)

# np.set_printoptions(threshold=np.inf) # help to print full array value in numpy
nclasses = 5

def convert(str):
    return int("".join(re.findall("\d*", str)))

def hist_norm(imgs, BWM=0., GWM=255.):
    # BWM = 0; % a minimum desired level
    # GWM = 255; % a maximum desired level
    
    # slices, row, col
    nslices, insz_h, insz_w = imgs.shape[0], imgs.shape[1], imgs.shape[2]	 
    # print(nslices, insz_h, insz_w)
    converted_data = np.reshape(imgs, (1, nslices*insz_h*insz_w))
    converted_data = converted_data.astype('float32')
    gmin = np.min(converted_data); # a minimum level of original 3D MRI data
    gmax = np.max(converted_data); # a maximum level of original 3D MRI data
    # print (gmax)
    
    # Normalize between BWM and GWM
    converted_data = (GWM - BWM) * (converted_data - gmin) / (gmax - gmin) + BWM
           
    imgs_norm = np.reshape(converted_data, (nslices, insz_h, insz_w))
    return imgs_norm

def read_scans(file_path1, data_tr_test=False):
    # nfiles = len(file_path1)
    scan_idx = 0
    for name in file_path1:
        # print ('\t', name)
        file_scan = sitk.ReadImage(name) # (240, 240, 155) = (rows, cols, slices)
        # print(flairs_scan.GetSize())
        nda = sitk.GetArrayFromImage(file_scan) # convert to numpy array, (155, 240, 240)
        if data_tr_test:
            nda = hist_norm(nda, 0., 255.)
            # print(nda.shape)
        
        if scan_idx == 0:
            nda_sum = nda            
            # print(gt_sum.shape)
        else:
            # nda_sum = np.append(nda_sum, nda, axis=0)
            nda_sum = np.concatenate((nda_sum, nda), axis=0) # faster
            # print(nda_sum.shape)
        
        scan_idx += 1
    
    # print(nda_sum.shape)
    return nda_sum

def resize_data(imgs_train, imgs_label):
    # prepare data for CNNs with the softmax activation
    nslices = 0
    for n in range(imgs_train.shape[0]):
        label_temp = imgs_label[n] # imgs_label[n][:][:]
        edges = sobel(label_temp)
        # print(label_temp.shape)
        c = np.count_nonzero(edges)
        # print(c)   
        if c > 499:
#==============================================================================
#             train_temp = imgs_train[n] # imgs_train[n, :, :]
#             train_temp2 = train_temp > 0
#             label_img = label(train_temp2, connectivity=train_temp.ndim)
#             props = regionprops(label_img)            
#             if len(props) > 1:
#                 area_max = 0
#                 idx = -1
#                 for prop in props:
#                     if prop.area > area_max:
#                         area_max = prop.area
#                         idx += 1
#                         # print(idx)
#             else:
#                 idx = 0
#             
#             min_row, min_col, max_row, max_col = props[idx].bbox
#             train_roi = train_temp[(min_row-2):(max_row+2), (min_col-2):(max_col+2)]
#             label_roi = label_temp[(min_row-2):(max_row+2), (min_col-2):(max_col+2)]
#             train_resz = resize(train_roi, (256, 256), preserve_range=True, mode='reflect')
#             label_resz = resize(label_roi, (64, 64), preserve_range=True, mode='reflect')
#             label_resz = label_resz.round()
#             
#             # train_resz2 = np.reshape(train_resz, (1, 256, 256))
#             train_resz2 = train_resz
#             train_resz2 = train_resz2[np.newaxis, ...]
#             
#             label_resz2 = np.reshape(label_resz, 64*64).astype('int32') 
#==============================================================================
            train_resz2 = imgs_train[n] # keep the original size of data
            train_resz2 = train_resz2[np.newaxis, ...]
            
            label_resz = label_temp
            label_resz2 = np.reshape(label_resz, 240*240).astype('int32')
            label_resz2 = to_categorical(label_resz2, nclasses)
                       
            label_resz2 = label_resz2[np.newaxis, ...] # 1, 240*240, nclasses
            if nslices == 0:
                # flair_sum = np.asarray([flair_resz]) same as np.reshape(label_resz, (1, 64, 64))
                # gt_sum = np.asarray([gt_resz])
                data_sum = train_resz2
                label_sum = label_resz2
            else:                
                data_sum = np.concatenate((data_sum, train_resz2), axis=0) # faster                
                label_sum = np.concatenate((label_sum, label_resz2), axis=0)
            
            nslices += 1
    # print(train_sum.shape)
    return data_sum, label_sum

def resize_data_2(imgs_train, imgs_label):
    # prepare data for CNNs with the sigmoid activation
    nslices = 0
    for n in range(imgs_train.shape[0]):
        label_temp = imgs_label[n] # imgs_label[n][:][:]
        # print(label_temp.shape)
        c = np.count_nonzero(label_temp)
        # print(c)   
        if c > 319:
            train_temp = imgs_train[n] # imgs_train[n, :, :]
            train_temp2 = train_temp > 0
            label_img = label(train_temp2, connectivity=train_temp.ndim)
            props = regionprops(label_img)
            # area_max = props[0].area
            # print(props[0].bbox[1])
            if len(props) > 1:
                area_max = 0
                idx = -1
                for prop in props:
                    if prop.area > area_max:
                        area_max = prop.area
                        idx += 1
                        # print(idx)
            else:
                idx = 0
            
            min_row, min_col, max_row, max_col = props[idx].bbox
            train_roi = train_temp[(min_row-2):(max_row+2), (min_col-2):(max_col+2)]
            label_roi = label_temp[(min_row-2):(max_row+2), (min_col-2):(max_col+2)]
            train_resz = resize(train_roi, (256, 256), preserve_range=True, mode='reflect')
            label_resz = resize(label_roi, (64, 64), preserve_range=True, mode='reflect')
            label_resz = label_resz.round() # value of label is 0-4 int
            
            # train_resz2 = np.reshape(train_resz, (1, 256, 256))            
            train_resz2 = train_resz[np.newaxis, ...]            
            label_resz2 = label_resz[np.newaxis, ...] # np.reshape(label_resz, (1, 64, 64))
            if nslices == 0:
                # flair_sum = np.asarray([flair_resz]) same as np.reshape(label_resz, (1, 64, 64))
                # gt_sum = np.asarray([gt_resz])
                data_sum = train_resz2
                label_sum = label_resz2
            else:                
                data_sum = np.concatenate((data_sum, train_resz2), axis=0) # faster                
                label_sum = np.concatenate((label_sum, label_resz2), axis=0)
            
            nslices += 1
    # print(train_sum.shape)
    return data_sum, label_sum

def create_train_data(type_train='softmax'):
    flairs = glob('D:\mhafiles\HGG\*\*Flair*\*Flair*.mha')
    gts = glob('D:\mhafiles\HGG\*\*OT*\*OT*.mha')
    
    flairs.sort(key=convert)
    gts.sort(key=convert)
    # print(flairs)
    # nfiles = len(flairs)
    # print(get_size)
    flair_sum = read_scans(flairs, True)
    print(flair_sum.shape)
    gt_sum = read_scans(gts)
    print(gt_sum.shape)
    
    if type_train == 'softmax':
        print('Resizing training data for the softmax activation...')
        flair_train, gt_train = resize_data(flair_sum, gt_sum)
        print(flair_train.shape)
        print(gt_train.shape)
        
        # np.save('imgs_train.npy', flair_train)
        # np.save('imgs_label_train.npy', gt_train)
        np.save('imgs_train_unet.npy', flair_train)
        np.save('imgs_label_train_unet.npy', gt_train)
        print('Saving all training data to .npy files done.')          
    elif type_train == 'sigmoid':
        print('Resizing training data for the sigmoid activation...')
        flair_train_2, gt_train_2 = resize_data_2(flair_sum, gt_sum)
        print(flair_train_2.shape)
        print(gt_train_2.shape)
        
        np.save('imgs_train_sigmoid.npy', flair_train_2)
        np.save('imgs_label_train_sigmoid.npy', gt_train_2)
        print('Saving all training data to .npy files done.')  
    else:
        print('Cannot save type of data as you want')    

def load_train_data(type_train='softmax'):
    if type_train == 'softmax':
        # imgs_train = np.load('imgs_train.npy')
        # imgs_label = np.load('imgs_label_train.npy')
        imgs_train = np.load('imgs_train_unet.npy')
        imgs_label = np.load('imgs_label_train_unet.npy')
    elif type_train == 'sigmoid':
        imgs_train = np.load('imgs_train_sigmoid.npy')
        imgs_label = np.load('imgs_label_train_sigmoid.npy')
    else:
        print('No type of data as you want')
        
    return imgs_train, imgs_label
    
def create_test_data(type_train='softmax'):
    # y_train = np.random.randint(10, size=(20, 1))
    # print(y_train.shape)
    # tests = glob('D:\mhafiles\HGG_Flair_5.mha')
    # gts = glob('D:\mhafiles\HGG_OT_5.mha')
    tests = glob('D:\mhafiles\HGG_Flair_pat111.mha')
    gts = glob('D:\mhafiles\HGG_OT_pat111.mha')
    # tests = glob('D:\mhafiles\H_LGG_Test\*\*Flair*\*54205.mha') # HGG
    # print(tests)
    test_sum = read_scans(tests, True)
    print(test_sum.shape)
    gt_sum = read_scans(gts)
    
    if type_train == 'softmax':
        print('Resizing testing data for the softmax activation...')
        flair_test, gt_test = resize_data(test_sum, gt_sum)
        print(flair_test.shape)
        print(gt_test.shape)
        
        np.save('imgs_test_unet_2.npy', flair_test)
        np.save('imgs_label_test_unet_2.npy', gt_test)
        # np.save('imgs_test_unet.npy', flair_test)
        # np.save('imgs_label_test_unet.npy', gt_test)
        print('Saving testing data to .npy files done.')
    elif type_train == 'sigmoid':
        print('Resizing testing data for the sigmoid activation...')
        flair_test_2, gt_test_2 = resize_data_2(test_sum, gt_sum)
        print(flair_test_2.shape)
        print(gt_test_2.shape)
        
        np.save('imgs_test_sigmoid.npy', flair_test_2)
        np.save('imgs_label_test_sigmoid.npy', gt_test_2)
        print('Saving testing data to .npy files done.')
    else:
        print('Cannot save type of data as you want')     
    
def load_test_data(type_train='softmax'):
    if type_train == 'softmax':
        # imgs_test = np.load('imgs_test_unet.npy')
        # imgs_label_test = np.load('imgs_label_test_unet.npy')
        imgs_test = np.load('imgs_test_unet_2.npy')
        imgs_label_test = np.load('imgs_label_test_unet_2.npy')
    elif type_train == 'sigmoid':
        imgs_test = np.load('imgs_test_sigmoid.npy')
        imgs_label_test = np.load('imgs_label_test_sigmoid.npy')
    else:
        print('No type of data as you want')
        
    return imgs_test, imgs_label_test

if __name__ == '__main__':
    # create_train_data('softmax')
    create_test_data('softmax')    