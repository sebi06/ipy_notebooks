#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time, os, sys
import mxnet as mx
import matplotlib.pyplot as plt
import glob
import sys
from cellpose import plot, transforms
#sys.path.insert(0,'/github/cellpose/')
from cellpose import models, utils
from aicsimageio import AICSImage, imread
#import imgfileutils as imf

# check if GPU working, and if so use it
use_gpu = utils.use_gpu()

print('Use GPU: ', use_gpu)


if use_gpu:
    device = mx.gpu()
else:
    device = mx.cpu()


# model_type='cyto' or model_type='nuclei'
model = models.Cellpose(device, model_type='nuclei')

# define CHANNELS to run segementation on
# grayscale=0, R=1, G=2, B=3
# channels = [cytoplasm, nucleus]
# if NUCLEUS channel does not exist, set the second channel to 0

# IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
# channels = [0,0] # IF YOU HAVE GRAYSCALE
# channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
# channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

ns = 1

#channels = [[2,3], [2,3]]
#channels = [[0,0], [0,0]]
channels = ns * [0, 0]


#filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate/testwell96.czi'
filename = r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\testwell96.czi'

# Get an AICSImage object
img = AICSImage(filename)

imgs = []

for s in range(ns):
#for s in range(img.size_s):
    
    data = img.get_image_data("YX", S=s, T=0, Z=0, C=0)
    data = data[200:700, 300:800]
    imgs.append(data)

img.close()

#nimg = len(imgs)

# if rescale is set to None, the size of the cells is estimated on a per image basis
# if you want to set `rescale` yourself (recommended), set it to 30. / average_cell_diameter
#masks, flows, styles, diams = model.eval(imgs, rescale=None, channels=channels)
masks, flows, styles, diams = model.eval(imgs, rescale=30, channels=channels)

for idx in range(len(imgs)):
    img = transforms.reshape(imgs[idx], channels[idx])
    t1 = img[0, :, :]
    img_rgb = plot.rgb_image(img)
    maski = masks[idx]
    flowi = flows[idx][0]

    
    fig = plt.figure(figsize=(16,12))
    # can save images (set save_dir=None if not)
    plot.show_segmentation(fig, img_rgb, maski, flowi)
    
    """    
    # display the result
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    ax[0, 0].imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    ax[0, 1].imshow(maski)
    ax[1, 0].imshow(flowi)
    #ax[1, 1].imshow(image_label_overlay)
    #ax[1, 1].imshow(labels)
    """
    
    plt.tight_layout()
    plt.show()






