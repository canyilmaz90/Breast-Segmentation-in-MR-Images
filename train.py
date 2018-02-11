#%% Initialization
from __future__ import print_function

import os
import cv2

import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dropout, merge
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from skimage.measure import label 

from data import load_train_data, load_test_data
from data import image_cols, image_rows
from prepare import preprocess, getLargestCC
from prepare import img_rows, img_cols

#K.set_image_data_format('channels_last')  # TF dimension ordering in this code
K.set_image_dim_ordering('th')
save_path = 'save/'


def unet():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    drop2 = Dropout(0.25)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.25)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 1)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = merge([drop3,up7], mode = 'concat', concat_axis = 1)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = merge([drop2,up8], mode = 'concat', concat_axis = 1)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 1)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(3, (1, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(input=[inputs], output=[conv10])

#    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=2.5e-5), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_predict(x_train, y_train, x_test):
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = unet()
    model_checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(x_train, y_train, batch_size=6, epochs=30, verbose=1, shuffle=True,
              validation_split=0.28,
              callbacks=[model_checkpoint])

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    y_test = model.predict(x_test, verbose=1)
    np.save(save_path + 'imgs_mask_test.npy', y_test)
    return y_test
    
#    for image, image_id in zip(imgs_mask_test, imgs_id_test):
#        image = (image[:, :, 0] * 255.).astype(np.uint8)
#        imsave(os.path.join(outdir, str(image_id) + '_pred.png'), image)


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
        img = imgs_mask_test[i].reshape((320, 320, 1))
        img = postprocess(img) * 255
        cv2.imwrite(os.path.join(outdir, 'segmask_' + str(imgs_id[i]) + '.png'), img)
    print('-' * 50)
    print('segmented images are retrieved')
    print('-' * 50)



#%% Data Preparation
if __name__ == '__main__':
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()
    imgs_train = imgs_train.astype('float32')
    imgs_train /= 255.
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.
    
    print('-'*30)
    print('Resizing train data ...')
    print('-'*30)
    imgs_train = preprocess(imgs_train)
#    imgs_trainn = imgs_train.reshape((42, 96, 96))
    imgs_mask_train = preprocess(imgs_mask_train)
    
    imgs_mask_train[imgs_mask_train <= 0.2] = 0;
    imgs_mask_train[(imgs_mask_train > 0.2) & (imgs_mask_train <= 0.6) ] = 0.5;
    imgs_mask_train[imgs_mask_train > 0.6] = 1;
    
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = imgs_test.astype('float32')
    imgs_test /= 255.
    imgs_test = preprocess(imgs_test)
    imgs_testt = imgs_test.reshape((17, 96, 96))
#%% Reshape masks into categorical structure
    print('-'*30)
    print('Creating categorical mask data ...')
    print('-'*30)
    trainmask = np.zeros((42, 1, 96, 96)).astype('float32')
    trainmask[imgs_mask_train == 1] = 1
    trainmask[imgs_mask_train == 0.5] = 2
    imgs_mask_train = keras.utils.to_categorical(trainmask, num_classes=3)
    
    onlybackground = imgs_mask_train[:, :, :, :, 0].reshape((42, 96, 96)).astype('float32')
    onlybreast = imgs_mask_train[:, :, :, :, 1].reshape((42, 96, 96)).astype('float32')
    onlychest = imgs_mask_train[:, :, :, :, 2].reshape((42, 96, 96)).astype('float32')
    onlychest = getLargestCC(onlychest)
    
    imgs_mask_train = np.zeros((42,3,96,96)).astype('float32')
    imgs_mask_train[:,0,:,:] = onlybackground
    imgs_mask_train[:,1,:,:] = onlybreast
    imgs_mask_train[:,2,:,:] = onlychest
#%% Normalization    
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std
    
    imgs_test -= mean
    imgs_test /= std

#%% Train and Predict    
    imgs_mask_test = train_predict(imgs_train, onlybreast.reshape((42, 1, 96, 96)), imgs_test)
#    model = unet()
#    model.load_weights('weights.hdf5')
#    imgs_mask_test = model.predict(imgs_test, verbose=1)    
#    imgs_mask_test = np.load(save_path + 'imgs_mask_test.npy')
    testmask = imgs_mask_test.reshape((17, 96, 96))
#    testmask[testmask1 < 0.5] = 0
#    testmask[testmask1 >= 0.5] = 1
