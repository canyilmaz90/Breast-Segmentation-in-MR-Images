from __future__ import print_function

import os
import os.path
import numpy as np
import cv2
from scipy.ndimage.interpolation import zoom
from skimage.transform import resize


data_path = 'mri/'
img_rows = 96
img_cols = 96


def prep_dicomdata(patient):
    print('*'*50)
    print('*'*50)
    print('Patient: ' + patient)
    print('-'*50)
    patientdir = os.path.join(data_path, patient + '/')
    outdir = os.path.join(patientdir, 'resampled/')
    images = os.listdir(patientdir + '/original/')
    total = len(images) // 5
    
    img_sample = cv2.imread(os.path.join(patientdir + '/original/', images[0]), cv2.IMREAD_GRAYSCALE)
    img_sample = np.array([img_sample])
    rows = img_sample.shape[1]
    cols = img_sample.shape[2]

    imgs = np.ndarray((total, rows, cols), dtype=np.uint8)
    
    print('-'*50)
    print('loading images...')
    print('-'*50)
    for i in range(total):
        img = cv2.imread(os.path.join(patientdir + '/original/', images[i*5 + 4]), cv2.IMREAD_GRAYSCALE)
        img = np.array([img])
        imgs[i] = img

    if not os.path.exists(outdir):
        os.mkdir(outdir)    
    np.save(outdir + 'imgs.npy', imgs)
    print('loading done.')
    
    if rows == 400:
        spacing = np.array([2.0, 0.8, 0.8])
    elif rows == 432:
        spacing = np.array([2.0, 0.78703701496124, 0.78703701496124])
    elif rows == 512:
        spacing = np.array([2.0, 0.78125, 0.78125])
    else:
        raise ValueError("This shape is unknown. Modify code again!")
    
    new_spacing =  np.array([1.0, 1.0, 1.0])
    newimg = resample(imgs, spacing, new_spacing, 2)
    print('-'*50)
    print('resampling done.')
    print('-'*50)
    
    if newimg.shape[1] < 336:
        wr = (336 - newimg.shape[1]) // 2
        wc = (336 - newimg.shape[2]) // 2
        padwidth = ((0, 0), (wr, wr), (wc, wc))
        newimg = np.pad(newimg, padwidth, mode='constant', constant_values=0)
    elif newimg.shape[1] > 336:
        wr = (newimg.shape[1] - 336) // 2
        wc = (newimg.shape[2] - 336) // 2
        newimgtemp = np.ndarray((newimg.shape[0], 336, 336), dtype=np.uint8)
        for i in range(newimg.shape[0]):
            temp = newimg[i].reshape(newimg.shape[1], newimg.shape[2])
            newimgtemp[i] = temp[wr : wr + 336, wc : wc + 336]
        newimg = newimgtemp
    
    for i in range(newimg.shape[0]):
        cv2.imwrite(outdir + str(i+1) + '.png', newimg[i])
    print('new images are created.')


def resample(imgs, spacing, new_spacing, order=2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg
    else:
        raise ValueError('wrong shape')
        

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], 1, img_rows, img_cols))
    imgs_p = imgs_p.astype('float32')
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (1, img_rows, img_cols), preserve_range=True)

    return imgs_p


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
#%% --------Preparing Dicom Data-----------
    patients = os.listdir(data_path)
    for patient in patients:
        prep_dicomdata(patient)