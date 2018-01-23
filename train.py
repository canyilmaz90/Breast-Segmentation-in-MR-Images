from __future__ import print_function

import os
import cv2

import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Input, concatenate, Dropout, merge
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from skimage.measure import label 

from data import load_train_data, load_test_data
from data import image_cols, image_rows
from prepare import preprocess
from prepare import img_rows, img_cols

#K.set_image_data_format('channels_last')  # TF dimension ordering in this code
K.set_image_dim_ordering('th')
save_path = 'save/'


def unet():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    up6 = merge([Conv2D(512, 2, 2, border_mode='same')(UpSampling2D(size=(2, 2))(conv5)), conv4], mode='concat', concat_axis=1)
#    concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = merge([Conv2D(512, 2, 2, border_mode='same')(UpSampling2D(size=(2, 2))(conv6)), conv3], mode='concat', concat_axis=1)
#    concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = merge([Conv2D(512, 2, 2, border_mode='same')(UpSampling2D(size=(2, 2))(conv7)), conv2], mode='concat', concat_axis=1)
#    concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = merge([Conv2D(512, 2, 2, border_mode='same')(UpSampling2D(size=(2, 2))(conv8)), conv1], mode='concat', concat_axis=1)
#    concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(3, (1, 1), activation='softmax')(conv9)

    model = Model(input=[inputs], output=[conv10])

#    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-2), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train(imgs_train,imgs_mask_train):
#    mean = np.mean(imgs_train)  # mean for data centering
#    std = np.std(imgs_train)  # std for data normalization
#
#    imgs_train -= mean
#    imgs_train /= std

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = unet()
    model_checkpoint = ModelCheckpoint(save_path + 'weights.hdf5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=6, epochs=20, verbose=1, shuffle=True,
              validation_split=0.28,
              callbacks=[model_checkpoint])


def predict():
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test /= 255.
#    imgs_test -= mean
#    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model = unet()
    model.load_weights(save_path + 'weights.hdf5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save(save_path + 'imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    
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

def getLargestCC(imarray):
    imarray2 = np.zeros(imarray.shape)
    for im in range(42):
        image = imarray[im]
        image = image.astype('uint8')
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
        sizes = stats[:, -1]
        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]
        img2 = np.zeros(output.shape)
        img2[output == max_label] = 1
        imarray2[im] = img2
    return imarray2

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
    imgs_mask_train = preprocess(imgs_mask_train)
    
    imgs_mask_train[imgs_mask_train <= 0.2] = 0;
    imgs_mask_train[(imgs_mask_train > 0.2) & (imgs_mask_train <= 0.6) ] = 0.5;
    imgs_mask_train[imgs_mask_train > 0.6] = 1;
    
    print('-'*30)
    print('Creating categorical mask data ...')
    print('-'*30)
    trainmask = np.zeros((42,1,96,96))
    trainmask[imgs_mask_train == 1] = 1
    trainmask[imgs_mask_train == 0.5] = 2
    imgs_mask_train = keras.utils.to_categorical(trainmask, num_classes=3)
    
    asd1 = imgs_mask_train[0:387072, 0]
    asd1 = asd1.reshape((42,96,96))
    asd2 = imgs_mask_train[0:387072, 1]
    asd2 = asd2.reshape((42,96,96))
    asd3 = imgs_mask_train[0:387072, 2]
    asd3 = asd3.reshape((42,96,96))
    asd3 = getLargestCC(asd3)
    
    imgs_mask_train = np.zeros((42,3,96,96))
    imgs_mask_train[:,0,:,:] = asd1
    imgs_mask_train[:,1,:,:] = asd2
    imgs_mask_train[:,2,:,:] = asd3
    
    train(imgs_train,imgs_mask_train)
    predict()
#    
    imgs_mask_test = np.load(save_path + 'imgs_mask_test.npy')
    testmask1 = imgs_mask_test[:,0,:,:]
    testmask1 = testmask1.reshape((42, 96, 96))
    testmask2 = imgs_mask_test[:,1,:,:]
    testmask2 = testmask1.reshape((42, 96, 96))
    testmask3 = imgs_mask_test[:,2,:,:]
    testmask3 = testmask1.reshape((42, 96, 96))
    