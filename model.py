from __future__ import print_function

import os
import cv2
import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, Dropout, merge
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
#K.set_image_data_format('channels_last')  # TF dimension ordering in this code
K.set_image_dim_ordering('th')
save_path = 'save/'


def unet():
    from run_file import img_rows, img_cols
    inputs = Input((1, img_rows, img_cols))
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
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
    model.compile(optimizer=SGD(lr=4e-3), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train(x_train, y_train):
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = unet()
    model_checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    history = model.fit(x_train, y_train, batch_size=70, epochs=200, verbose=1, shuffle=True,
              validation_split=0.25,
              callbacks=[model_checkpoint])
    return history

def predict(x_test):
    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    model = unet()
    model.load_weights('weights.hdf5')
    y_test = model.predict(x_test, verbose=1)
    np.save(save_path + 'imgs_mask_test.npy', y_test)
    return y_test
    
#    for image, image_id in zip(imgs_mask_test, imgs_id_test):
#        image = (image[:, :, 0] * 255.).astype(np.uint8)
#        imsave(os.path.join(outdir, str(image_id) + '_pred.png'), image)


#if __name__ == '__main__':
#    x_train, y_train = load_train_data()
#    train(x_train, y_train)
#    x_test, imgs_id = load_test_data()
#    y_test = predict(x_test)