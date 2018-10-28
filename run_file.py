#%%
from __future__ import print_function

import os
import cv2
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

from data import load_train_data, load_test_data
from data import image_cols, image_rows
#from prepare import preprocess, getLargestCC
from model import unet, train, predict

data_path = 'raw/'
save_path = 'save/'

img_rows = 128
img_cols = 128

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], 1, img_rows, img_cols))
    imgs_p = imgs_p.astype('float32')
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (1, img_rows, img_cols), preserve_range=True)

    return imgs_p

def postprocess(img):
    img = img.astype('float32')
    img = cv2.threshold(img, 0.5, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
    img = cv2.resize(img, (image_cols, image_rows))
    return img

def retrieve_imgs():
    outdir = 'output'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    imgs_mask_test = np.load(save_path + 'imgs_mask_test.npy')
    imgs_id = np.load(save_path + 'imgs_id_test.npy')
    num_imgs = imgs_mask_test.shape[0]
    for i in range(num_imgs):
        img = imgs_mask_test[i].reshape((img_rows, img_cols, 1))
        img = postprocess(img) * 255
        cv2.imwrite(os.path.join(outdir, 'segmask_' + str(imgs_id[i]) + '.png'), img)
    print('-' * 50)
    print('segmented images are retrieved')
    print('-' * 50)

if __name__ == '__main__':
# Load Training Data and Train the Model
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    x_train, y_train = load_train_data()
    x_train = x_train.astype('float32')
    x_train /= 255.
    x_train = preprocess(x_train)
    y_train = y_train.astype('float32')
    y_train /= 255.
    y_train = preprocess(y_train)
# Batch Normalization
    for i in range(x_train.shape[0]):
        imean = np.mean(x_train[i])
        x_train[i] = np.power(x_train[i],imean/0.1) # gamma correction
    mean = np.mean(x_train)  # mean for data centering
    std = np.std(x_train)  # std for data normalization
    x_train -= mean
    x_train /= std
    
    history = train(x_train, y_train)
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

#%% Load Test Data and Predict the Test Masks
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    x_test, imgs_id = load_test_data()
    x_test = x_test.astype('float32')
    x_test /= 255.
    x_test = preprocess(x_test)
    
# Batch Normalization
    for i in range(x_test.shape[0]):
        imean = np.mean(x_test[i])
        x_test[i] = np.power(x_test[i],imean/0.1) # gamma correction
    mean = np.mean(x_test)  # mean for data centering
    std = np.std(x_test)  # std for data normalization
    x_test -= mean
    x_test /= std
    
    predict(x_test)
    retrieve_imgs()