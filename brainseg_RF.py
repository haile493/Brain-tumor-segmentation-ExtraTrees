# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 09:41:50 2017

@author: THANHHAI
"""
# using the pre-trained models to extract features from FLAIR, T2, T1c, T1 MRI sequences
# and then train Extremely Randomized Trees classifier

from __future__ import print_function
import os
# force tensorflow to use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
# from skimage import data, util
# from skimage.measure import label, regionprops
from skimage import io, color
from skimage.filters import sobel
# from skimage.morphology import remove_small_objects
from skimage.morphology import opening, square
# from skimage.transform import resize
import SimpleITK as sitk
from matplotlib import pyplot as plt
# from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from glob import glob
from sklearn.externals import joblib
import re
import gc
from timeit import default_timer as timer
from skimage.util.montage import montage2d # for making 3d montages from 2D images
# from skimage.util.shape import view_as_windows
# from skimage.util import montage

# ----------------------------------------------------------------------------
import tensorflow as tf
# import keras
from keras.models import Model, model_from_json
# from keras.layers import Dense, Dropout, Activation, Flatten, Merge, Reshape
# from keras.layers import Input, MaxoutDense
# from keras.layers import Conv2D, MaxPooling2D, AtrousConv2D
# from keras.optimizers import SGD
# from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import ModelCheckpoint
from keras import backend as K

# ----------------------------------------------------------------------------
# an argument to your Session's config arguments, it might help a little, 
# though it won't release memory, it just allows growth at the cost of some memory efficiency
K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
# with tf.Session(config = config) as s:
sess = tf.Session(config = config)
K.set_session(sess)
# ----------------------------------------------------------------------------

def convert(str):
    return int("".join(re.findall("\d*", str)))

def dice_score_full(y_pred, y_true):
    # dice coef of entire tumor
# =============================================================================
#     y_true_f = y_true.flatten() 
#     y_pred_f = y_pred.flatten() 
#     c_true = np.count_nonzero(y_true_f)
#     c_pred = np.count_nonzero(y_pred_f)
# =============================================================================
# =============================================================================
#     true_pos = np.float(np.sum(y_true * y_pred))
#     union = np.float(np.sum(y_true) + np.sum(y_pred))
#     if union == 0:
#         whole_tumor = 0
#     else:
#         whole_tumor = 2.0 * true_pos / union
# =============================================================================
    y_true_b = np.asarray(y_true).astype(np.bool)
    y_pred_b = np.asarray(y_pred).astype(np.bool)
    union = y_pred_b.sum() + y_true_b.sum()
    intersection = np.logical_and(y_pred_b, y_true_b)
    if union == 0:
        whole_tumor = 0
    else:
        whole_tumor = (2.0 * intersection.sum()) / union
    
    # dice coef of enhancing tumor
    enhan_gt = np.argwhere(y_true == 4)
    gt_a, seg_a = [], [] # classification of
    for i in enhan_gt:
        gt_a.append(y_true[i[0]][i[1]])
        seg_a.append(y_pred[i[0]][i[1]])
    gta = np.array(gt_a)
    sega = np.array(seg_a)
    if len(enhan_gt) > 0:
        enhan_tumor = float(len(np.argwhere(gta == sega))) / float(len(enhan_gt))
    else:
        enhan_tumor = 0
    
    # dice coef core tumor
    noenhan_gt = np.argwhere(y_true == 3)
    necrosis_gt = np.argwhere(y_true == 1)
    live_tumor_gt = np.append(enhan_gt, noenhan_gt, axis = 0)
    core_gt = np.append(live_tumor_gt, necrosis_gt, axis = 0)
    gt_core, seg_core = [], []
    for i in core_gt:
        gt_core.append(y_true[i[0]][i[1]])
        seg_core.append(y_pred[i[0]][i[1]])
    gtcore = np.array(gt_core)
    segcore = np.array(seg_core)
    if len(core_gt) > 0:
        core_tumor = float(len(np.argwhere(gtcore == segcore))) / float(len(core_gt))
    else:
        core_tumor = 0
    
    return whole_tumor, enhan_tumor, core_tumor

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

# A deep learning model integrating FCNNs and CRFs for brain tumor seg
# Xiaomei Zhao, et. al.
def intensity_norm(imgs, sigma0=0., gray_val0=0., first_scan=True):
    # slices, row, col
    nslices, insz_h, insz_w = imgs.shape[0], imgs.shape[1], imgs.shape[2]
    converted_data = np.reshape(imgs, (1, nslices*insz_h*insz_w))
    converted_data = converted_data.astype('float32')
    
    BWM = 0.    # a minimum desired level
    GWM = 255.  # a maximum desired level
    gmin = np.min(converted_data) # a minimum level of original 3D MRI data
    gmax = np.max(converted_data) # a maximum level of original 3D MRI data
    # print (gmax)
    
    # Normalize between BWM and GWM
    converted_data = (GWM - BWM) * (converted_data - gmin) / (gmax - gmin) + BWM
            
    hist, _ = np.histogram(converted_data, bins=256)
    hist[0] = 0
    gray_val = np.argmax(hist) # gray level of highest histogram bin
    
    no_voxels = converted_data > 0 # find positions of the value greater than 0 in data
    N = no_voxels.sum() # total number of pixels is greater than 0
    converted_data[no_voxels] -= gray_val
    sum_val = np.square(converted_data[no_voxels]).sum()
    sigma = np.sqrt(sum_val/N)
    converted_data[no_voxels] /= sigma
    
    # if not first_scan:
    if first_scan: 
        converted_data[no_voxels] *= sigma
        converted_data[no_voxels] += gray_val
    else:
        converted_data[no_voxels] *= sigma0
        converted_data[no_voxels] += gray_val0
        
# =============================================================================
#     no_data1 = converted_data < 0
#     converted_data[no_data1] = 0.
#     no_data2 = converted_data > 255
#     converted_data[no_data2] = 255.
# =============================================================================
    
    converted_data[converted_data<0.] = 0.
    converted_data[converted_data>255.] = 255.
    
    # print(sigma)
    
    imgs_normed = np.reshape(converted_data, (nslices, insz_h, insz_w))
    
    return imgs_normed, sigma, gray_val

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

# apply the intensity normalization method
def read_scans_IN(file_path1, MRI_type=0):
    # nfiles = len(file_path1)
    if MRI_type == 0:
        sigma, graylevel = np.load('imgs_flair_sig_gray_N4.npy')
        print("sigma: %.3f, gray level: %i" % (sigma, graylevel))
    elif MRI_type == 1:
        sigma, graylevel = np.load('imgs_t1c_sig_gray_N4.npy')
        print("sigma: %.3f, gray level: %i" % (sigma, graylevel))
    elif MRI_type == 2:
        sigma, graylevel = np.load('imgs_t2_sig_gray_N4.npy')
        print("sigma: %.3f, gray level: %i" % (sigma, graylevel))
    else:
        sigma, graylevel = np.load('imgs_t1_sig_gray_N4.npy')
        print("sigma: %.3f, gray level: %i" % (sigma, graylevel))
        
    scan_idx = 0
    for name in file_path1:
        # print ('\t', name)
        file_scan = sitk.ReadImage(name) # (240, 240, 155) = (rows, cols, slices)
        # print(flairs_scan.GetSize())
        nda = sitk.GetArrayFromImage(file_scan) # convert to numpy array, (155, 240, 240)        
        nda, _, _ = intensity_norm(nda, sigma, graylevel, False)
        # print(nda.shape)
        
        if scan_idx == 0:
            nda_sum = nda            
            # print(gt_sum.shape)
        else:            
            nda_sum = np.concatenate((nda_sum, nda), axis=0) # faster
            # print(nda_sum.shape)
        
        scan_idx += 1
    
    # print(nda_sum.shape)
    return nda_sum

def read_scans_IN_LG(file_path1, MRI_type=0):
    # nfiles = len(file_path1)
    if MRI_type == 0:
        sigma, graylevel = np.load('imgs_flair_sig_gray_N4_LG.npy')
        print("sigma: %.3f, gray level: %i" % (sigma, graylevel))
    elif MRI_type == 1:
        sigma, graylevel = np.load('imgs_t1c_sig_gray_N4_LG.npy')
        print("sigma: %.3f, gray level: %i" % (sigma, graylevel))
    elif MRI_type == 2:
        sigma, graylevel = np.load('imgs_t2_sig_gray_N4_LG.npy')
        print("sigma: %.3f, gray level: %i" % (sigma, graylevel))
    else:
        sigma, graylevel = np.load('imgs_t1_sig_gray_N4_LG.npy')
        print("sigma: %.3f, gray level: %i" % (sigma, graylevel))
        
    scan_idx = 0
    for name in file_path1:
        # print ('\t', name)
        file_scan = sitk.ReadImage(name) # (240, 240, 155) = (rows, cols, slices)
        # print(flairs_scan.GetSize())
        nda = sitk.GetArrayFromImage(file_scan) # convert to numpy array, (155, 240, 240)        
        nda, _, _ = intensity_norm(nda, sigma, graylevel, False)
        # print(nda.shape)
        
        if scan_idx == 0:
            nda_sum = nda            
            # print(gt_sum.shape)
        else:            
            nda_sum = np.concatenate((nda_sum, nda), axis=0) # faster
            # print(nda_sum.shape)
        
        scan_idx += 1
    
    # print(nda_sum.shape)
    return nda_sum

def resize_data(imgs_train1, imgs_train2, imgs_train3, imgs_train4, imgs_label, data_type):
    # prepare data for CNNs with the softmax activation
    if data_type == 'HG':
        T = 199
    else:
        T = 99
    
    nslices = 0
    for n in range(imgs_train1.shape[0]):
        label_temp = imgs_label[n] # imgs_label[n][:][:]
        edges = sobel(label_temp)
        # print(label_temp.shape)
        c = np.count_nonzero(edges)
        # print(c)   
        if c > T:
            train_resz1 = imgs_train1[n] # keep the original size of data
            train_resz2 = imgs_train2[n]
            train_resz3 = imgs_train3[n]
            train_resz4 = imgs_train4[n]
            train_resz1 = train_resz1[np.newaxis, ...]
            train_resz2 = train_resz2[np.newaxis, ...]
            train_resz3 = train_resz3[np.newaxis, ...]
            train_resz4 = train_resz4[np.newaxis, ...]
            
            label_resz = label_temp.astype('int')
            label_resz[label_resz==3] = 1            
            # label_resz2 = np.reshape(label_resz, 240*240).astype('float32')        
            label_resz2 = np.reshape(label_resz, 240*240)
            
            if nslices == 0:
                # flair_sum = np.asarray([flair_resz]) same as np.reshape(label_resz, (1, 64, 64))
                # gt_sum = np.asarray([gt_resz])
                data_sum1 = train_resz1
                data_sum2 = train_resz2
                data_sum3 = train_resz3
                data_sum4 = train_resz4
                label_sum = label_resz2
            else:                
                data_sum1 = np.concatenate((data_sum1, train_resz1), axis=0) # faster  
                data_sum2 = np.concatenate((data_sum2, train_resz2), axis=0)
                data_sum3 = np.concatenate((data_sum3, train_resz3), axis=0)
                data_sum4 = np.concatenate((data_sum4, train_resz4), axis=0)
                label_sum = np.concatenate((label_sum, label_resz2), axis=0)
            
            nslices += 1
    # print(train_sum.shape)
    return data_sum1, data_sum2, data_sum3, data_sum4, label_sum

def resize_data_test(imgs_train1, imgs_train2, imgs_train3, imgs_train4, imgs_label):
    # prepare data for CNNs with the softmax activation
    nslices = 0
    for n in range(imgs_train1.shape[0]):
        label_temp = imgs_label[n] # imgs_label[n][:][:]
        edges = sobel(label_temp)
        # print(label_temp.shape)
        c = np.count_nonzero(edges)
        # print(c)   
        if c >= 0:
            train_resz1 = imgs_train1[n] # keep the original size of data
            train_resz2 = imgs_train2[n]
            train_resz3 = imgs_train3[n]
            train_resz4 = imgs_train4[n]
            train_resz1 = train_resz1[np.newaxis, ...]
            train_resz2 = train_resz2[np.newaxis, ...]
            train_resz3 = train_resz3[np.newaxis, ...]
            train_resz4 = train_resz4[np.newaxis, ...]
            
            label_resz = label_temp.astype('int')
            label_resz[label_resz==3] = 1
            # keep the original shape of test data
            # label_resz2 = label_resz
            label_resz = label_resz[np.newaxis, ...]                                  
            
            if nslices == 0:
                # flair_sum = np.asarray([flair_resz]) same as np.reshape(label_resz, (1, 64, 64))
                # gt_sum = np.asarray([gt_resz])
                data_sum1 = train_resz1
                data_sum2 = train_resz2
                data_sum3 = train_resz3
                data_sum4 = train_resz4
                label_sum = label_resz
            else:                
                data_sum1 = np.concatenate((data_sum1, train_resz1), axis=0) # faster  
                data_sum2 = np.concatenate((data_sum2, train_resz2), axis=0)
                data_sum3 = np.concatenate((data_sum3, train_resz3), axis=0)
                data_sum4 = np.concatenate((data_sum4, train_resz4), axis=0)
                label_sum = np.concatenate((label_sum, label_resz), axis=0)
            
            nslices += 1
    
    return data_sum1, data_sum2, data_sum3, data_sum4, label_sum

def load_trained_model():   
    # load json and create model
    json_file = open('cnn_BRATs_unet.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("cnn_BRATs_unet.h5")
    print("Loaded UNET model from disk")
    
    return loaded_model

def save_model(model, name):
    # name = 'filename.pkl'
    filepath = os.path.join('D:/mhafiles/model/', name) 
    # filepath = r"D:\mhafiles\model\" + name
    joblib.dump(model, filepath, compress=1) 
    print('Saved ExtraTrees model.')

def load_model(name):
    filepath = os.path.join('D:/mhafiles/model/', name) 
    model= joblib.load(filepath) 
    
    return model

def show_img(imgs_pred, imgs_label):
    for n in range(imgs_pred.shape[0]):
        img_pred = imgs_pred[n].astype('float32')
        img_label = imgs_label[n].astype('float32')
        
        # dice = dice_score(imgs_pred[n], imgs_label[n])
        # print("Dice score: %.3f" %dice)
        whole_tumor, enhan_tumor, core_tumor = dice_score_full(img_pred, img_label)
        print("Whole tumor: %.3f, Enhancing tumor: %.3f, Core: %.3f" % (whole_tumor, enhan_tumor, core_tumor))
                
        # img_1 = cvt2color_img(img_pred)
        # img_2 = cvt2color_img(img_label)
        cnames = ['red', 'green', 'blue', 'yellow']
        img_1 = color.label2rgb(img_pred, colors=cnames, bg_label=0) 
        img_2 = color.label2rgb(img_label, colors=cnames, bg_label=0)
        show_2Dimg(img_1, img_2)
        
        # show_2Dimg(imgs_pred[n], imgs_label[n])
                
def show_2Dimg(img_1, img_2):
    fig, axes = plt.subplots(ncols=2)
    ax = axes.ravel()
    ax[0].imshow(img_1, cmap=plt.cm.gray)
    ax[1].imshow(img_2, cmap=plt.cm.gray)
    plt.show()
    
def show_2Dimg_2(img_pred, img_true, CvtImg=True): 
    whole_tumor, enhan_tumor, core_tumor = dice_score_full(img_pred, img_true)
    print("Whole tumor: %.3f, Enhancing tumor: %.3f, Core: %.3f" % (whole_tumor, enhan_tumor, core_tumor))    
    
    if CvtImg:
        img_1 = cvt2color_img(img_pred)
        # img_22 = cvt2color_img(img_2)
    else:
        img_1 = img_pred 
        # img_22 = img_2
    
    img_2 = cvt2color_img(img_true)  
    
    fig, axes = plt.subplots(ncols=2)
    ax = axes.ravel()
    ax[0].imshow(img_1, cmap=plt.cm.gray)
    ax[1].imshow(img_2, cmap=plt.cm.gray)
    plt.show()
    
def show_2Dimg_3(data): 
    img_src, img_pred, img_true = data  
    
    img_1 = cvt2color_img(img_pred.astype('float32'))
    img_2 = cvt2color_img(img_true.astype('float32'))  
    
    fig, axes = plt.subplots(ncols=3)
    ax = axes.ravel()
    ax[0].imshow(img_src, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[1].imshow(img_1, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[2].imshow(img_2, cmap=plt.cm.gray)
    ax[2].axis('off')
    plt.show()
    
def show_img_overlap(data):
    img_src, img_pred, img_true = data
    # whole_tumor, enhan_tumor, core_tumor = dice_score_full(img_pred, img_true)
    # print("Whole tumor: %.3f, Core: %.3f, Enhancing tumor: %.3f" % (whole_tumor, core_tumor, enhan_tumor))    
        
    label_pred = img_pred.astype('int16')
    label_true = img_true.astype('int16')
       
    img_dst = color.gray2rgb(img_src)
    img_dst /= 255.
    # img_dst2 = img_dst
    
    # check colors in cvt2color_img function    
    # colors_label = [(1, 0.2, 0.2), (0.35, 0.75, 0.25), (0, 0.25, 0.9), (1, 1, 0.25)]
    # overlay_pred = color.label2rgb(label_pred, image=img_dst, colors=colors_label, bg_label=0) 
    # overlay_true = color.label2rgb(label_true, image=img_dst, colors=colors_label, bg_label=0)
    
    cnames = ['red', 'green', 'blue', 'yellow']    
    overlay_pred = color.label2rgb(label_pred, image=img_dst, colors=cnames, bg_label=0) 
    # cnames2 = ['red', 'green', 'blue', 'yellow'] # ['red', 'green', 'yellow', 'blue']
    overlay_true = color.label2rgb(label_true, image=img_dst, colors=cnames, bg_label=0)
    
    fig, axes = plt.subplots(ncols=3)
    ax = axes.ravel()
    ax[0].imshow(img_dst, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[1].imshow(overlay_pred, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[2].imshow(overlay_true, cmap=plt.cm.gray)
    ax[2].axis('off')
    plt.show()
        
def cvt2color_img(img_src):
# =============================================================================
#     ones = np.argwhere(img_src == 1) # class 1/necrosis
#     twos = np.argwhere(img_src == 2) # class 2/edema
#     threes = np.argwhere(img_src == 3) # class 3/non-enhancing tumor
#     fours = np.argwhere(img_src == 4) # class 4/enhancing tumor
# =============================================================================
    
    img_dst = color.gray2rgb(img_src)
    red_multiplier = [1, 0.2, 0.2] # class 1/necrosis    
    green_multiplier = [0.35, 0.75, 0.25] # class 2/edema
    blue_multiplier = [0, 0.25, 0.9] # class 3/non-enhancing tumor
    yellow_multiplier = [1, 1, 0.25] # class 4/enhancing tumor
    img_dst[img_src == 1] = red_multiplier
    img_dst[img_src == 2] = green_multiplier
    img_dst[img_src == 3] = blue_multiplier
    img_dst[img_src == 4] = yellow_multiplier
    
# =============================================================================
#     # change colors of segmented classes
#     for i in range(len(ones)):
#         img_dst[ones[i][0]][ones[i][1]] = red_multiplier
#     for i in range(len(twos)):
#         img_dst[twos[i][0]][twos[i][1]] = green_multiplier
#     for i in range(len(threes)):
#         img_dst[threes[i][0]][threes[i][1]] = blue_multiplier
#     for i in range(len(fours)):
#         img_dst[fours[i][0]][fours[i][1]] = yellow_multiplier
# =============================================================================
        
    return img_dst

def show_montage_filters(imgs_src):
    # show the features that extract from MRI using UNET
    # change position of axis from (240,240,64) to (64,240,240) 
    imgs_show = np.moveaxis(imgs_src, -1, 0)
    # print(imgs_tmp.shape)    
    slice_montage = montage2d(imgs_show)

    fig, (ax1) = plt.subplots(1,1, figsize = (12, 12))    
    ax1.imshow(slice_montage, cmap = 'gray')
    ax1.axis('off')
    ax1.set_title('Features')  
    plt.show()

def test_model(data_test, gt_test, modelEXT, data_type, save_file=False):
    if data_type == 'HG':
        imgs_sum = np.load('D:\mhafiles\Data\sum_test_3types.npy')
    elif data_type == 'LG':
        imgs_sum = np.load('D:\mhafiles\Data\sum_test_3types_LG.npy')
        
    imgs_flair = imgs_sum[:, :, :, 0]
    if not save_file:
        wtumor = []
        ctumor = []
        etumor = []
    
    for n in range(data_test.shape[0]):
        img_test = data_test[n]
        img_label = gt_test[n]
        img_flair = imgs_flair[n]
        # print(img_flair.shape)
        
        img_pred = modelEXT.predict(img_test)
        img_pred = np.reshape(img_pred, [240, 240])
        img_label = np.reshape(img_label, [240, 240])        
        # print(train_pred.shape)
        
        # img_pred = remove_small_objects(img_pred.astype('int16'), min_size=10, connectivity=2)
        img_pred = opening(img_pred.astype('int16'), square(3))
# =============================================================================
#             ones = np.argwhere(img_pred == 1)
#             print(len(ones))
#             twos = np.argwhere(img_pred == 2)
#             print(len(twos))
#             threes = np.argwhere(img_pred == 3)
#             print(len(threes))
#             fours = np.argwhere(img_pred == 4)
#             print(len(fours))
# =============================================================================
            # img_pred[img_pred==3] = 1            
            # img_pred_post = opening(img_pred.astype('int16'), square(3))
        
        # show_2Dimg(train_pred, label_roi)
        # show_2Dimg_2(img_pred, img_label)        
        if not save_file:
            print('Slice: %i' %n)
            whole_tumor, enhan_tumor, core_tumor = dice_score_full(img_pred, img_label)
            print("Whole tumor: %.3f, Core: %.3f, Enhancing tumor: %.3f" % (whole_tumor, core_tumor, enhan_tumor))
            if whole_tumor > 0:
                wtumor.append(whole_tumor)
            if core_tumor > 0:
                ctumor.append(core_tumor)
            if enhan_tumor > 0:
                etumor.append(enhan_tumor)
            # show_img_overlap([img_flair, img_pred, img_label])
            show_2Dimg_3([img_flair, img_pred, img_label])
        else:
            img_pred2 = img_pred[np.newaxis, ...]
            if n == 0:
                imgs_pred = img_pred2
            else:
                imgs_pred = np.concatenate((imgs_pred, img_pred2), axis=0)
    
    if save_file:
        print('Saving the predicted result to mha file ...')        
        # convert numpy data to SimpleITK data, (155, 240, 240) to (240, 240, 155)
        imgs_sitk = sitk.GetImageFromArray(imgs_pred.astype('int16')) 
        if data_type == 'HG':            
            sitk.WriteImage(imgs_sitk, 'D:\mhafiles\Eval\VSD.Seg_HG_0002.54518.mha')
        elif data_type == 'LG': 
            sitk.WriteImage(imgs_sitk, 'D:\mhafiles\Eval\VSD.Seg_LG_0002.54638.mha')
    else:
        if len(wtumor) > 0:
            print('Whole tumor: mean = %s, std = %s' %(np.mean(wtumor), np.std(wtumor)))
        else:
            print('Whole tumor: mean = 0.0 std = 0.0')
        if len(ctumor) > 0: 
            print('Core tumor: mean = %s, std = %s' %(np.mean(ctumor), np.std(ctumor)))
        else:
            print('Core tumor: mean = 0.0, std = 0.0')
        if len(etumor) > 0:
            print('Enhancing tumor: mean = %s, std = %s' %(np.mean(etumor), np.std(etumor)))
        else:
            print('Enhancing tumor: mean = 0.0, std = 0.0')

def Extra_RF(X, y):
    # Extremely Randomized Trees / Extra Trees Classification
    num_trees = 50 # 100
    mdepth = 15 # 50
    # max_features = 1.0
    # model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
    model = ExtraTreesClassifier(n_estimators=num_trees, max_depth=mdepth)
    # model_val = cross_val_score(model, X, y)
    # print(model_val.mean())
    results = model.fit(X, y)
        
    return results

def cvt_data_training(imgs_1, imgs_2, imgs_3, imgs_4):
    if (imgs_1.ndim == imgs_2.ndim) and (imgs_2.ndim == imgs_3.ndim):
        for n in range(imgs_1.shape[0]):
            img_1 = imgs_1[n]
            img_2 = imgs_2[n]
            img_3 = imgs_3[n]
            img_4 = imgs_4[n]
            nrows, ncols, nfils = img_1.shape
            # print(img_1.shape)
            img_1_resh = np.reshape(img_1, (nrows*ncols, nfils))
            img_2_resh = np.reshape(img_2, (nrows*ncols, nfils))
            img_3_resh = np.reshape(img_3, (nrows*ncols, nfils))
            img_4_resh = np.reshape(img_4, (nrows*ncols, nfils))
            img_sum = np.concatenate((img_1_resh, img_2_resh, img_3_resh, img_4_resh), axis=-1)
            # img_sum = np.concatenate((img_sum, img_3_resh), axis=1)
            # img_sum = np.concatenate((img_sum, img_4_resh), axis=1)
            # 64*4 = 256 features
            # print(img_sum.shape)
            if n == 0:
                nda_sum = img_sum
            else:
                nda_sum = np.concatenate((nda_sum, img_sum), axis=0)
    
    return nda_sum

def cvt_data_testing(imgs_1, imgs_2, imgs_3, imgs_4, gts_test):
    # imgs_1, imgs_2, imgs_3, imgs_4, gts_test = data
    if (imgs_1.ndim == imgs_2.ndim) and (imgs_2.ndim == imgs_3.ndim):
        for n in range(imgs_1.shape[0]):
            img_1 = imgs_1[n]
            img_2 = imgs_2[n]
            img_3 = imgs_3[n]
            img_4 = imgs_4[n]
            gt_test = gts_test[n]
            nrows, ncols, nfils = img_1.shape
            # print(img_1.shape)
            img_1_resh = np.reshape(img_1, (nrows*ncols, nfils)) # 240*240, 64
            img_2_resh = np.reshape(img_2, (nrows*ncols, nfils))
            img_3_resh = np.reshape(img_3, (nrows*ncols, nfils))
            img_4_resh = np.reshape(img_4, (nrows*ncols, nfils))
            img_sum = np.concatenate((img_1_resh, img_2_resh, img_3_resh, img_4_resh), axis=-1) # 240*240, 64*4
            # img_sum = np.concatenate((img_sum, img_3_resh), axis=1)
            # img_sum = np.concatenate((img_sum, img_4_resh), axis=1)
            img_sum = img_sum[np.newaxis, ...]
            
            gt_img = np.reshape(gt_test, nrows*ncols)
            gt_img = gt_img[np.newaxis, ...]
            # print(img_sum.shape)
            if n == 0:
                nda_sum = img_sum
                gt_sum = gt_img
            else:
                nda_sum = np.concatenate((nda_sum, img_sum), axis=0)
                gt_sum = np.concatenate((gt_sum, gt_img), axis=0)
    
    return nda_sum, gt_sum

def data_concat(data):
    imgs_1, imgs_2, imgs_3 = data
    imgs_1 = imgs_1[..., np.newaxis]
    imgs_2 = imgs_2[..., np.newaxis]
    imgs_3 = imgs_3[..., np.newaxis]
    
    data_sum = np.concatenate((imgs_1, imgs_2, imgs_3), axis=-1)
    # data_sum = np.concatenate((data_sum), axis=-1)
    
    return data_sum

def create_train_data(data_type='HG'):
    if data_type == 'HG':
# =============================================================================
#         flairs = glob(r'D:\mhafiles\RF\N4ITK_HGG_Flair_1.mha')
#         T1cs = glob(r'D:\mhafiles\RF\N4ITK_HGG_T1c_1.mha')    
#         T2s = glob(r'D:\mhafiles\RF\N4ITK_HGG_T2_1.mha')
#         T1s = glob(r'D:\mhafiles\RF\N4ITK_HGG_T1_1.mha')
#         gts = glob('D:\mhafiles\RF\HGG_OT_1.mha')
# =============================================================================
        flairs = glob('D:\mhafiles\RF\HGG_Flair_1.mha')
        T1cs = glob('D:\mhafiles\RF\HGG_T1c_1.mha')    
        T2s = glob('D:\mhafiles\RF\HGG_T2_1.mha')
        T1s = glob('D:\mhafiles\RF\HGG_T1_1.mha')
        gts = glob('D:\mhafiles\RF\HGG_OT_1.mha')
    elif data_type == 'LG':
# =============================================================================
#         flairs = glob(r'D:\mhafiles\RF\N4ITK_LGG_Flair_1.mha')
#         T1cs = glob(r'D:\mhafiles\RF\N4ITK_LGG_T1c_1.mha')    
#         T2s = glob(r'D:\mhafiles\RF\N4ITK_LGG_T2_1.mha')
#         T1s = glob(r'D:\mhafiles\RF\N4ITK_LGG_T1_1.mha')
#         gts = glob('D:\mhafiles\RF\LGG_OT_1.mha')
# =============================================================================
        flairs = glob('D:\mhafiles\RF\LGG_Flair_1.mha')
        T1cs = glob('D:\mhafiles\RF\LGG_T1c_1.mha')    
        T2s = glob('D:\mhafiles\RF\LGG_T2_1.mha')
        T1s = glob('D:\mhafiles\RF\LGG_T1_1.mha')
        gts = glob('D:\mhafiles\RF\LGG_OT_1.mha')
    else:
        print('Error!!!')
    
    flairs.sort(key=convert)
    T1cs.sort(key=convert)
    T2s.sort(key=convert)
    T1s.sort(key=convert)
    gts.sort(key=convert)
    
    flair_sum = read_scans(flairs, True)
    print(flair_sum.shape)
    T1c_sum = read_scans(T1cs, True)
    print(T1c_sum.shape)
    T2_sum = read_scans(T2s, True)
    print(T2_sum.shape)
    T1_sum = read_scans(T1s, True)
    print(T1_sum.shape)
    gt_sum = read_scans(gts)
    print(gt_sum.shape)
    
# =============================================================================
#     flair_sum = read_scans_IN(flairs, 0)
#     print(flair_sum.shape)
#     T1c_sum = read_scans_IN(T1cs, 1)
#     print(T1c_sum.shape)
#     T2_sum = read_scans_IN(T2s, 2)
#     print(T2_sum.shape)
#     T1_sum = read_scans_IN(T1s, 3)
#     print(T1_sum.shape)
#     gt_sum = read_scans(gts)
#     print(gt_sum.shape)
# =============================================================================
    
    print('Collecting slices for training data ...')
    flair_train, T1c_train, T2_train, T1_train, gt_train = resize_data(flair_sum, T1c_sum, 
                                                             T2_sum, T1_sum, gt_sum, data_type)
    print(flair_train.shape)
    print(T1c_train.shape)
    print(T2_train.shape)
    print(T1_train.shape)
    print(gt_train.shape)
        
    # np.save('imgs_flair.npy', flair_train)
    # np.save('imgs_T1c.npy', T1c_train)
    # np.save('imgs_T2.npy', T2_train)
    # np.save('imgs_label.npy', gt_train)
    # print('Saving all training data to .npy files done.')
    
    print('Extracting features for training ...')
    flair_features, T1c_features, T2_features, T1_features = feature_extraction(flair_train, 
                                                                                T1c_train,
                                                                                T2_train, 
                                                                                T1_train)         
    print(flair_features.shape)
    print(T1c_features.shape)
    print(T2_features.shape)
    print(T1_features.shape)
    
    # show_montage_filters(flair_features[0])
    # show_montage_filters(T1c_features[0])
    # show_montage_filters(T2_features[0])
    # show_montage_filters(T1_features[0])
    
    print('Convert data for training ExtraTrees classfier ...')
    data_sum = cvt_data_training(flair_features, T1c_features, T2_features, T1_features)
    print(data_sum.shape)
    
    if data_type == 'HG':
        print('Saving total training features of HGG to 1 file ...')
        np.save('D:\mhafiles\Data\sum_train_features.npy', data_sum)
        np.save('D:\mhafiles\Data\sum_train_gt.npy', gt_train)
    elif data_type == 'LG':
        print('Saving total training features of LGG to 1 file ...')
        np.save('D:\mhafiles\Data\sum_train_features_LG.npy', data_sum)
        np.save('D:\mhafiles\Data\sum_train_gt_LG.npy', gt_train)
    else:
        print('No data type for saving ...')
    
    print('Done!')
    
    for i in range(20):
        gc.collect()

def create_test_data(data_type='HG'):
    if data_type == 'HG':        
# =============================================================================
#         flairs_test = glob('D:\mhafiles\HGG_Flair_2.mha')
#         t1cs_test = glob('D:\mhafiles\HGG_T1c_2.mha')
#         t2s_test = glob('D:\mhafiles\HGG_T2_2.mha')
#         t1s_test = glob('D:\mhafiles\HGG_T1_2.mha')
#         gts_test = glob('D:\mhafiles\HGG_OT_2.mha')
# =============================================================================
        # BRATS 2015
# =============================================================================
#         flairs_test = glob(r'D:\mhafiles\B2015_Training\HG\*2013_pat0010*\*Flair*\N4ITK_*Flair*.mha')
#         t1cs_test = glob(r'D:\mhafiles\B2015_Training\HG\*2013_pat0010*\*T1c*\N4ITK_*T1c*.mha')
#         t2s_test = glob(r'D:\mhafiles\B2015_Training\HG\*2013_pat0010*\*T2*\N4ITK_*T2*.mha')
#         t1s_test = glob(r'D:\mhafiles\B2015_Training\HG\*2013_pat0010*\*T1.*\N4ITK_*T1.*.mha')
#         gts_test = glob('D:\mhafiles\B2015_Training\HG\*2013_pat0010*\*OT*\*OT*.mha') 
# =============================================================================
        flairs_test = glob('D:\mhafiles\B2015_Training\HG\*2013_pat0002*\*Flair*\VSD.*Flair*.mha')
        t1cs_test = glob('D:\mhafiles\B2015_Training\HG\*2013_pat0002*\*T1c*\VSD.*T1c*.mha')
        t2s_test = glob('D:\mhafiles\B2015_Training\HG\*2013_pat0002*\*T2*\VSD.*T2*.mha')
        t1s_test = glob('D:\mhafiles\B2015_Training\HG\*2013_pat0002*\*T1.*\VSD.*T1.*.mha')
        gts_test = glob('D:\mhafiles\B2015_Training\HG\*2013_pat0002*\*OT*\*OT*.mha') 
    elif data_type == 'LG':
# =============================================================================
#         flairs_test = glob('D:\mhafiles\LGG_Flair_2.mha')
#         t1cs_test = glob('D:\mhafiles\LGG_T1c_2.mha')
#         t2s_test = glob('D:\mhafiles\LGG_T2_2.mha')
#         t1s_test = glob('D:\mhafiles\LGG_T1_2.mha')
#         gts_test = glob('D:\mhafiles\LGG_OT_2.mha')
# =============================================================================
        # BRATS 2015
# =============================================================================
#         flairs_test = glob(r'D:\mhafiles\B2015_Training\LG\*2013_pat0002*\*Flair*\N4ITK_*Flair*.mha')
#         t1cs_test = glob(r'D:\mhafiles\B2015_Training\LG\*2013_pat0002*\*T1c*\N4ITK_*T1c*.mha')
#         t2s_test = glob(r'D:\mhafiles\B2015_Training\LG\*2013_pat0002*\*T2*\N4ITK_*T2*.mha')
#         t1s_test = glob(r'D:\mhafiles\B2015_Training\LG\*2013_pat0002*\*T1.*\N4ITK_*T1.*.mha')
#         gts_test = glob('D:\mhafiles\B2015_Training\LG\*2013_pat0002*\*OT*\*OT*.mha')     
# =============================================================================
        flairs_test = glob('D:\mhafiles\B2015_Training\LG\*2013_pat0002*\*Flair*\VSD.*Flair*.mha')
        t1cs_test = glob('D:\mhafiles\B2015_Training\LG\*2013_pat0002*\*T1c*\VSD.*T1c*.mha')
        t2s_test = glob('D:\mhafiles\B2015_Training\LG\*2013_pat0002*\*T2*\VSD.*T2*.mha')
        t1s_test = glob('D:\mhafiles\B2015_Training\LG\*2013_pat0002*\*T1.*\VSD.*T1.*.mha')
        gts_test = glob('D:\mhafiles\B2015_Training\LG\*2013_pat0002*\*OT*\*OT*.mha') 
    else:
        print('Error!!!')
    
    flairs_test.sort(key=convert)
    print(flairs_test)
    t1cs_test.sort(key=convert)
    t2s_test.sort(key=convert)
    t1s_test.sort(key=convert)
    gts_test.sort(key=convert)
    
    flair_sum_test = read_scans(flairs_test, True)
    print(flair_sum_test.shape)
    t1c_sum_test = read_scans(t1cs_test, True) 
    print(t1c_sum_test.shape)
    t2_sum_test = read_scans(t2s_test, True) 
    print(t2_sum_test.shape)
    t1_sum_test = read_scans(t1s_test, True) 
    print(t1_sum_test.shape)
    gt_sum_test = read_scans(gts_test)
    print(gt_sum_test.shape)
    
# =============================================================================
#     if data_type == 'HG':
#         flair_sum_test = read_scans_IN(flairs_test, 0)
#         print(flair_sum_test.shape)
#         t1c_sum_test = read_scans_IN(t1cs_test, 1) 
#         print(t1c_sum_test.shape)
#         t2_sum_test = read_scans_IN(t2s_test, 2) 
#         print(t2_sum_test.shape)
#         t1_sum_test = read_scans_IN(t1s_test, 3) 
#         print(t1_sum_test.shape)
#     elif data_type == 'LG':
#         flair_sum_test = read_scans_IN_LG(flairs_test, 0)
#         print(flair_sum_test.shape)
#         t1c_sum_test = read_scans_IN_LG(t1cs_test, 1) 
#         print(t1c_sum_test.shape)
#         t2_sum_test = read_scans_IN_LG(t2s_test, 2) 
#         print(t2_sum_test.shape)
#         t1_sum_test = read_scans_IN_LG(t1s_test, 3) 
#         print(t1_sum_test.shape)
#     
#     gt_sum_test = read_scans(gts_test)
#     print(gt_sum_test.shape)
# =============================================================================
    
    print('Collecting slices for test data ...')
    flair_train, t1c_train, t2_train, t1_train, gt_train = resize_data_test(flair_sum_test, t1c_sum_test, 
                                                             t2_sum_test, t1_sum_test, gt_sum_test)
    print(flair_train.shape)
    print(t1c_train.shape)
    print(t2_train.shape)
    print(t1_train.shape)
    print(gt_train.shape)    
    
    print('Extracting features for testing ...')
    flair_features, t1c_features, t2_features, t1_features = feature_extraction(flair_train, t1c_train, 
                                                                                t2_train, t1_train)         
    print(flair_features.shape)
    print(t1c_features.shape)
    print(t2_features.shape)
    print(t1_features.shape)
    
    print('Convert data for testing ExtraTrees classfier ...')
    data_sum, gt_test = cvt_data_testing(flair_features, t1c_features, 
                                         t2_features, t1_features, gt_train)
    print(data_sum.shape, gt_test.shape)
    
    if data_type == 'HG':
        print('Saving total test features of HGG to 1 file ...')
        np.save('D:\mhafiles\Data\sum_test_features.npy', data_sum) 
        np.save('D:\mhafiles\Data\sum_test_gt.npy', gt_test)
        
        data_3types = data_concat([flair_train, t1c_train, t2_train])
        np.save('D:\mhafiles\Data\sum_test_3types.npy', data_3types)   
    elif data_type == 'LG':
        print('Saving total test features of LGG to 1 file ...')
        np.save('D:\mhafiles\Data\sum_test_features_LG.npy', data_sum) 
        np.save('D:\mhafiles\Data\sum_test_gt_LG.npy', gt_test)   
        
        data_3types = data_concat([flair_train, t1c_train, t2_train])
        np.save('D:\mhafiles\Data\sum_test_3types_LG.npy', data_3types)   
    else:
        print('No data type for saving ...')
        
    print('Done!')
    
    for i in range(20):
        gc.collect()

def normalization_data(imgs_src):
    imgs_src = imgs_src[..., np.newaxis] 
    imgs_src = imgs_src.astype('float32') 
    imgs_src /= 255.
    
    mean = np.mean(imgs_src)  # mean for data centering
    std = np.std(imgs_src)  # std for data normalization
    # print('mean = %f, std = %f' %(mean, std))    
    
    imgs_src -= mean
    imgs_src /= std
        
    # minv = np.min(imgs_src)  # mean for data centering
    # maxv = np.max(imgs_src)  # std for data normalization
    # print('min = %f, max = %f' %(minv, maxv))   
    
    # print(imgs_src.shape)
    
    return imgs_src

def feature_extraction(imgs_1, imgs_2, imgs_3, imgs_4):
    # a pre-trained CNN (U-Net) could be used as a Feature Extractor    
    batch_size = 2
    
    imgs_11 = normalization_data(imgs_1)           
    imgs_22 = normalization_data(imgs_2)
    imgs_33 = normalization_data(imgs_3)
    imgs_44 = normalization_data(imgs_4)
        
    print('Loading the trained UNET model...')
    base_model = load_trained_model()
    # base_model.summary()
    
    # param 73792
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv2d_17').output)
    
    # param 36928
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv2d_18').output)
    
    print('Predicting model with data...')
    features_1 = model.predict(imgs_11, batch_size=batch_size, verbose=0)
    features_2 = model.predict(imgs_22, batch_size=batch_size, verbose=0)
    features_3 = model.predict(imgs_33, batch_size=batch_size, verbose=0)
    features_4 = model.predict(imgs_44, batch_size=batch_size, verbose=0)
    # prop = model.predict_generator(test_generator, steps=imgs_test.shape[0] // batch_size,
                                   # verbose=0)
    # print(prop.shape)
    
    # save_result_h5py(prop)
        
    # imgs_test_pred = convert_data_toimage(prop)
    # print(imgs_test_pred.shape)
    # maxv = np.max(imgs_test_pred)
    # print(maxv)
    # imgs_label = convert_data_toimage(imgs_label)
    # print(imgs_label.shape)
    # imgs_test2 = imgs_test * 255.
    # imgs_test2 = convert_data_toimage(imgs_test2)
    # print(imgs_test2.shape)
    
    # show_img(imgs_test_pred, imgs_label)
    # show_img(imgs_test2, imgs_test_pred)
    
    print('Feature extraction: Done!')
    
    return features_1, features_2, features_3, features_4
    
def load_and_get_features():
    print('Loading the selected slices ...')
    flair_train = np.load('imgs_flair.npy')
    T1c_train = np.load('imgs_T1c.npy')
    T2_train = np.load('imgs_T2.npy')
    print(flair_train.shape)
          
    flair_features, T1c_features, T2_features = feature_extraction(flair_train, T1c_train, T2_train)         
    print(flair_features.shape)
    print(T1c_features.shape)
    print(T2_features.shape)
     
    print('Saving the data of features ...')
    np.save('imgs_features.npy', [flair_features, T1c_features, T2_features])  
     
def load_and_cvt_data():
    print('Loading data ...')
    flair_features, T1c_features, T2_features = np.load('D:\mhafiles\Data\imgs_features.npy')
    
    print('Convert data for training ExtraTrees classfier ...')
    data_sum = cvt_data_training(flair_features, T1c_features, T2_features)
    
    print('Saving total features to 1 file ...')
    np.save('D:\mhafiles\Data\sum_features.npy', data_sum)      
        
    print('Done!')
    
def load_and_train_data(data_type='HG'):    
    if data_type == 'HG':
        print('Loading HGG data ...')
        imgs_train = np.load('D:\mhafiles\Data\sum_train_features.npy')    
        imgs_label = np.load('D:\mhafiles\Data\sum_train_gt.npy')
    elif data_type == 'LG':
        print('Loading LGG data ...')
        imgs_train = np.load('D:\mhafiles\Data\sum_train_features_LG.npy')    
        imgs_label = np.load('D:\mhafiles\Data\sum_train_gt_LG.npy')
    elif data_type == 'HLG':
        print('Loading HGG data ...')
        imgs_train = np.load('D:\mhafiles\Data\sum_train_features.npy')    
        imgs_label = np.load('D:\mhafiles\Data\sum_train_gt.npy')
        print('Loading LGG data ...')
        imgs_train2 = np.load('D:\mhafiles\Data\sum_train_features_LG.npy')    
        imgs_label2 = np.load('D:\mhafiles\Data\sum_train_gt_LG.npy')
        print('Combining HGG and LGG data for training ...')
        imgs_train = np.concatenate((imgs_train, imgs_train2), axis=0)
        imgs_label = np.concatenate((imgs_label, imgs_label2), axis=0)
    else:
        print('No data to load ...')
        
    print(imgs_train.shape, imgs_label.shape) 
    
    start = timer()
    print('Training data using ExtraTrees')
    model = Extra_RF(imgs_train, imgs_label)
    end = timer()
    print('Training time: %.2f (s)' %(end - start))
        
    if data_type == 'HG':
        print('Saving the trained model for HGG')
        save_model(model, 'trained_ExtraTrees_unet.pkl')
    elif data_type == 'LG':
        print('Saving the trained model for LGG')
        save_model(model, 'trained_ExtraTrees_unet_LG.pkl')
    elif data_type == 'HLG':
        print('Saving the trained model for HGG and LGG')
        save_model(model, 'trained_ExtraTrees_unet_HLG.pkl')        
        # set label 3 (non-enhancing tumor) to label 4 (enhancing tumor)
        # save_model(model, 'trained_ExtraTrees_unet_HLG_2.pkl') 
    
    # load_and_test_data(model)
    del model
    for i in range(20):
        gc.collect()
    
# def load_and_test_data(model):
def load_and_test_data(data_type='HG', model_type='HG'):  
    # start = timer()
    if data_type == 'HG':
        print('Loading test data and model of HGG ...')
        imgs_test = np.load('D:\mhafiles\Data\sum_test_features.npy')    
        imgs_label = np.load('D:\mhafiles\Data\sum_test_gt.npy')                
    elif data_type == 'LG':
        print('Loading test data and model of LGG ...')
        imgs_test = np.load('D:\mhafiles\Data\sum_test_features_LG.npy')    
        imgs_label = np.load('D:\mhafiles\Data\sum_test_gt_LG.npy')                
    else:
        print('No test data is loaded!')
        
    if model_type == 'HG':
        model = load_model('trained_ExtraTrees_unet.pkl')
    elif model_type == 'LG':
        model = load_model('trained_ExtraTrees_unet_LG.pkl')
    elif model_type == 'HLG':
        model = load_model('trained_ExtraTrees_unet_HLG.pkl')
        # model = load_model('trained_ExtraTrees_unet_HLG_2.pkl')
    else:
        print('No model is loaded!')
        
    print(imgs_test.shape, imgs_label.shape)
    
    print('Testing data using ExtraTrees')
    # model = load_model('trained_ExtraTrees_NoN4.pkl')
    # model = load_model('trained_ExtraTrees.pkl')
    # test_model(imgs_test, imgs_label, model, data_type, save_file=True)
    test_model(imgs_test, imgs_label, model, data_type) # default: no saving file
    
    # end = timer()
    # print('Test time: %.2f (s)' %(end - start))
    
    print('Testing done!')
    
    del model
    for i in range(20):
        gc.collect()
        
def clear_mem(n):
    for i in range(n):
        gc.collect()
         
if __name__ == '__main__':
    # create_train_data('HG')
    # create_test_data('HG')    
    # load_and_get_features()
    # load_and_cvt_data()
    # load_and_train_data('HG')
    # load_and_train_data('HLG')
    # load_and_test_data(data_type='HG', model_type='HG')
    # load_and_test_data(data_type='HG', model_type='HLG')
    
    # create_train_data('LG')
    # create_test_data('LG')    
    # load_and_train_data('LG')
    # load_and_test_data(data_type='LG', model_type='LG')
    load_and_test_data(data_type='LG', model_type='HLG')
    
    # clear_mem(20)
