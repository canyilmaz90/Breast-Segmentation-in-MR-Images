import os
import os.path
import numpy as np
import cv2
from scipy.ndimage.interpolation import zoom
from skimage.transform import resize
from data import image_cols, image_rows
data_path = 'C:/Users/canyi/Desktop/CHEN CUI YING/'

img_rows = 96
img_cols = 96


def prep_dicomdata():
    images = os.listdir(data_path)
    total = len(images) // 5

    imgs_1 = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_2 = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_3 = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_4 = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_5 = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    
    print('-'*50)
    print('loading images...')
    print('-'*50)
    for i in range(total):
        img_1 = cv2.imread(os.path.join(data_path, images[i*5]), cv2.IMREAD_GRAYSCALE)
        img_1 = np.array([img_1])
        imgs_1[i] = img_1
        img_2 = cv2.imread(os.path.join(data_path, images[i*5 + 1]), cv2.IMREAD_GRAYSCALE)
        img_2 = np.array([img_2])
        imgs_2[i] = img_2
        img_3 = cv2.imread(os.path.join(data_path, images[i*5 + 2]), cv2.IMREAD_GRAYSCALE)
        img_3 = np.array([img_3])
        imgs_3[i] = img_3
        img_4 = cv2.imread(os.path.join(data_path, images[i*5 + 3]), cv2.IMREAD_GRAYSCALE)
        img_4 = np.array([img_4])
        imgs_4[i] = img_4
        img_5 = cv2.imread(os.path.join(data_path, images[i*5 + 4]), cv2.IMREAD_GRAYSCALE)
        img_5 = np.array([img_5])
        imgs_5[i] = img_5
        
    np.save(os.path.join(data_path, 'resampled/') + 'imgs_1.npy', imgs_1)
    np.save(os.path.join(data_path, 'resampled/') + 'imgs_2.npy', imgs_2)
    np.save(os.path.join(data_path, 'resampled/') + 'imgs_3.npy', imgs_3)
    np.save(os.path.join(data_path, 'resampled/') + 'imgs_4.npy', imgs_4)
    np.save(os.path.join(data_path, 'resampled/') + 'imgs_5.npy', imgs_5)
    print('loading done.')



def resample(imgs, spacing, new_spacing, order=2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')
        

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], 1, img_rows, img_cols))
    imgs_p = imgs_p.astype('float32')
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (1, img_rows, img_cols), preserve_range=True)

    return imgs_p


if __name__ == '__main__':
#    --------Preparing Dicom Data-----------
    prep_dicomdata()
    imgs_1 = np.load(os.path.join(data_path, 'resampled/') + 'imgs_1.npy')
    cv2.imwrite(os.path.join(data_path, 'resampled/') + 'img_1.png', imgs_1[74])
    spacing = np.array([2.0, 0.8, 0.8])
    new_spacing =  np.array([1.0, 1.0, 1.0])
    imgsnew_1, truspc = resample(imgs_1, spacing, new_spacing, 2) 
#    --------------------------------------