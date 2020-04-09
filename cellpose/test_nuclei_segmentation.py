# -*- coding: utf-8 -*-

#################################################################
# File        : test_nuclei_segmentation.py
# Version     : 0.2
# Author      : czsrh
# Date        : 09.04.2020
# Institution : Carl Zeiss Microscopy GmbH
#
#
# Copyright (c) 2020 Carl Zeiss AG, Germany. All Rights Reserved.
#################################################################


# use_method = 'scikit'
# use_method = 'cellpose'
use_method = 'zentf'


import sys
import time
import os
from glob import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
# import imgfileutils as imf
from scipy import ndimage
from aicsimageio import AICSImage, imread
from skimage import exposure
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.segmentation import random_walker
from skimage import io, measure, segmentation
from skimage.filters import threshold_otsu, threshold_triangle, rank
from skimage.segmentation import clear_border
from skimage.color import label2rgb
from skimage.util import invert
from skimage.filters import median, gaussian
from skimage.morphology import closing, square
from skimage.morphology import remove_small_objects, remove_small_holes, disk
from skimage.measure import label, regionprops
from MightyMosaic import MightyMosaic

if use_method == 'cellpose':

    try:
        import mxnet
        from cellpose import plot, transforms
        from cellpose import models, utils

    except ImportError as error:
        # Output expected ImportErrors.
        print(error.__class__.__name__ + ": " + error.msg)

if use_method == 'zentf':

    try:
        # silence tensorflow output
        from silence_tensorflow import silence_tensorflow
        silence_tensorflow()
        import tensorflow as tf
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        print('TensorFlow Version : ', tf.version.GIT_VERSION, tf.__version__)
    except ImportError as error:
        # Output expected ImportErrors.
        print(error.__class__.__name__ + ": " + error.msg)

# select plotting backend
plt.switch_backend('Qt5Agg')
verbose = False


def set_device():

    # check if GPU working, and if so use it
    use_gpu = utils.use_gpu()
    print('Use GPU: ', use_gpu)

    if use_gpu:
        device = mxnet.gpu()
    else:
        device = mxnet.cpu()

    return device


def apply_watershed(binary, min_distance=30, footprint=3):

    # create distance map
    distance = ndimage.distance_transform_edt(binary)

    # dtermine local maxima
    local_maxi = peak_local_max(distance,
                                # min_distance=min_distance,
                                indices=False,
                                labels=binary,
                                footprint=np.ones((footprint, footprint)))

    # label maxima
    markers, num_features = ndimage.label(local_maxi)

    # apply watershed
    mask = watershed(-distance, markers, mask=binary, watershed_line=True)

    return mask


def autoThresholding(image2d,
                     method='triangle',
                     radius=10,
                     value=50):

    # calculate global Otsu threshold
    if method == 'global_otsu':
        thresh = threshold_otsu(image2d)

    # calculate local Otsu threshold
    if method == 'local_otsu':
        thresh = rank.otsu(image2d, disk(radius))

    if method == 'value_based':
        thresh = value

    if method == 'triangle':
        thresh = threshold_triangle(image2d)

    binary = image2d >= thresh

    return binary


def segment_nuclei_threshold(image2d,
                             filtermethod='median',
                             filtersize=3,
                             threshold='triangle',
                             split_ws=True,
                             mindist_ws=30):

    # filter image
    if filtermethod == 'median':
        image2d = median(image2d, selem=disk(filtersize))
    if filtermethod == 'gauss':
        image2d = gaussian(image2d, sigma=filtersize, mode='reflect')

    # threshold image and run marker-based watershed
    binary = autoThresholding(image2d, method='triangle')

    # apply watershed
    if split_ws:
        mask = apply_watershed(binary, min_distance=min_peakdist)
    if not split_ws:
        # label the objects
        mask, num_features = ndimage.label(binary)
        mask = mask.astype(np.int)

    return mask


def segment_nuclei_cellpose(image2d, model,
                            channels=[0, 0],
                            rescale=None):

    # define CHANNELS to run segmentation on
    # grayscale=0, R=1, G=2, B=3
    # channels = [cytoplasm, nucleus]
    # if NUCLEUS channel does not exist, set the second channel to 0

    # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
    # channels = [0,0] # IF YOU HAVE GRAYSCALE
    # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
    # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

    # get the mask for a single image
    masks, _, _, _ = model.eval([image2d], rescale=rescale, channels=channels)

    return masks[0]


def get_binary_from_prediction(prediction, classlabel=1):

    # Generate labels from one-hot encoded vectors
    prediction_labels = np.argmax(prediction, axis=-1)

    """
    # get the desired class
    background = 0
    nuclei = 1
    borders = 2
    """

    # extract desired class
    binary = np.where(prediction_labels == classlabel, 1, 0)

    return binary


def segment_zentf(image2d, model, classlabel):

    # segment a singe [X, Y] 2D image

    # add add batch dimension (at the front) and channel dimension (at the end)
    image2d = image2d[np.newaxis, ..., np.newaxis]

    # Run prediction - array shape must be [1, 1024, 1024, 1]
    prediction = model.predict(image2d)[0]  # Removes batch dimension

    # get the binary image with the labels
    binary = get_binary_from_prediction(prediction, classlabel=classlabel)

    return binary


def segment_zentf_tiling(image2d, model,
                         tilesize=1024,
                         classlabel=1,
                         overlap_factor=1):

    # create tile image
    image2d_tiled = MightyMosaic.from_array(image2d, (tilesize, tilesize),
                                            overlap_factor=overlap_factor,
                                            fill_mode='reflect')

    print('image2d_tiled shape : ', image2d_tiled.shape)
    # get number of tiles
    num_tiles = image2d_tiled.shape[0] * image2d_tiled.shape[1]
    print('Number of Tiles: ', num_tiles)

    # create array for the binary results
    binary_tiled = image2d_tiled

    ct = 0
    for n1 in range(image2d_tiled.shape[0]):
        for n2 in range(image2d_tiled.shape[1]):

            ct += 1
            print('Processing Tile : ', ct, ' Size : ', image2d_tiled[n1, n2, :, :].shape)

            # extract a tile
            tile = image2d_tiled[n1, n2, :, :]

            # get the binary from the prediction for a single tile
            binary_tile = segment_zentf(tile, model, classlabel=classlabel)

            # cats the result into the output array
            binary_tiled[n1, n2, :, :] = binary_tile

    # created fused binary and covert to int
    binary = binary_tiled.get_fusion().astype(int)

    return binary


def plot_results(image, mask, props, add_bbox=True):

    # create overlay image
    image_label_overlay = label2rgb(mask, image=image, bg_label=0)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    ax[0].imshow(image,
                 cmap=plt.cm.gray,
                 interpolation='nearest',
                 clim=[image.min(), image.max() * 0.5])

    ax[1].imshow(image_label_overlay,
                 clim=[image.min(), image.max() * 0.5])

    ax[0].set_title('Original', fontsize=12)
    ax[1].set_title('Masks', fontsize=12)

    if add_bbox:
        add_boundingbox(props, ax[0])

    plt.show()


def add_boundingbox(props, ax2plot):

    for index, row in props.iterrows():

        minr = row['bbox-0']
        minc = row['bbox-1']
        maxr = row['bbox-2']
        maxc = row['bbox-3']
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False,
                                  edgecolor='red',
                                  linewidth=1)
        ax2plot.add_patch(rect)


def cutout_subimage(image2d,
                    startx=0,
                    starty=0,
                    width=100,
                    height=200):

    image2d = image2d[starty: height, startx:width]

    return image2d


def add_padding(image2d, input_height=1024, input_width=1024):

    if len(image2d.shape) == 2:
        isrgb = False
        image2d = image2d[..., np.newaxis]
    else:
        isrgb = True

    padding_height = input_height - image2d.shape[0]
    padding_width = input_width - image2d.shape[1]
    padding_left, padding_right = padding_width // 2, padding_width - padding_width // 2
    padding_top, padding_bottom = padding_height // 2, padding_height - padding_height // 2

    image2d_padded = np.pad(image2d, ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)), 'reflect')

    if not isrgb:
        image2d_padded = np.squeeze(image2d_padded, axis=2)

    return image2d_padded, (padding_top, padding_bottom, padding_left, padding_right)


###############################################################################


# filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate/testwell96.czi'
# filename = r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\testwell96.czi'
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Castor\testwell96-A1_1024x1024_0.czi"
filename = r'segment_nuclei_CNN.czi'

# get AICSImageIO object using the python wrapper for libCZI
img = AICSImage(filename)
SizeS = img.size_s
SizeT = img.size_t
SizeZ = img.size_z

chindex = 0  # channel containing the nuclei
minsize = 100  # minimum object size
maxsize = 5000  # maximum object size

# define cutout size for subimage
cutimage = True
startx = 0
starty = 0
width = 500
height = 800

# define columns names for dataframe
cols = ['S', 'T', 'Z', 'C', 'Number']
objects = pd.DataFrame(columns=cols)

# optional dipslay of "some" results - empty list = no display
show_image = [0]

# for testing
SizeS = 1

# use watershed for splitting
use_ws = True
min_peakdist = 30

# load the ML model from cellpose when needed
if use_method == 'cellpose':

    # load cellpose model for cell nuclei using GPU or CPU
    print('Loading Cellpose Model ...')
    model = models.Cellpose(device=set_device(), model_type='nuclei')

    # define list of channels for cellpose
    # channels = SizeS * SizeT * SizeZ * [0, 0]
    channels = [0, 0]

# define model oath and load TF2 model when needed
if use_method == 'zentf':

    # Load the model
    MODEL_PATH = 'model_folder'
    model = tf.keras.models.load_model(MODEL_PATH)

    # Determine input shape required by the model and crop input image respectively
    tile_height, tile_width = model.signatures["serving_default"].inputs[0].shape[1:3]
    print('ZEN TF Model Tile Dimension : ', width, height)

###########################################################################

image_counter = 0
results = pd.DataFrame()

for s in range(SizeS):
    for t in range(SizeT):
        for z in range(SizeZ):

            values = {'S': s,
                      'T': t,
                      'Z': z,
                      'C': chindex,
                      'Number': 0}

            print('Analyzing S-T-Z-C: ', s, t, z, chindex)
            image2d = img.get_image_data("YX",
                                         S=s,
                                         T=t,
                                         Z=z,
                                         C=chindex)

            # cutout subimage
            if cutimage:
                image2d = cutout_subimage(image2d,
                                          startx=startx,
                                          starty=startx,
                                          width=width,
                                          height=height)

            if use_method == 'cellpose':
                # get the mask for the current image
                mask = segment_nuclei_cellpose(image2d, model,
                                               rescale=None,
                                               channels=channels)

            if use_method == 'scikit':
                mask = segment_nuclei_threshold(image2d,
                                                filtermethod='median',
                                                filtersize=3,
                                                threshold='triangle',
                                                split_ws=use_ws,
                                                mindist_ws=min_peakdist)

            if use_method == 'zentf':

                classlabel = 1

                # check if tiling is required
                if image2d.shape[0] > tile_height or image2d.shape[1] > tile_width:
                    binary = segment_zentf_tiling(image2d, model,
                                                  tilesize=tile_height,
                                                  classlabel=classlabel,
                                                  overlap_factor=2)

                elif image2d.shape[0] == tile_height and image2d.shape[1] == tile_width:
                    binary = segment_zentf(image2d, model, classlabel=classlabel)

                elif image2d.shape[0] < tile_height or image2d.shape[1] < tile_width:

                    # do padding
                    image2d_padded, pad = add_padding(image2d,
                                                      input_height=tile_height,
                                                      input_width=tile_width)

                    # run prediction on padded image
                    binary_padded = segment_zentf(image2d_padded, model, classlabel=classlabel)

                    # remove padding from result
                    binary = binary_padded[pad[0]:tile_height - pad[1], pad[2]:tile_width - pad[3]]

                # apply watershed
                if use_ws:
                    mask = apply_watershed(binary, min_distance=min_peakdist, footprint=35)

                if not use_ws:
                    # label the objects
                    mask, num_features = ndimage.label(binary)

            # clear the border
            mask = segmentation.clear_border(mask)

            # measure region properties
            to_measure = ('label',
                          'area',
                          'centroid',
                          'max_intensity',
                          'mean_intensity',
                          'min_intensity',
                          'bbox')

            props = pd.DataFrame(
                measure.regionprops_table(
                    mask,
                    intensity_image=image2d,
                    properties=to_measure
                )
            ).set_index('label')

            # filter by size
            props = props[(props['area'] >= minsize) & (props['area'] <= maxsize)]
            # props = [r for r in props if r.area >= minsize]

            props['S'] = s
            props['T'] = t
            props['Z'] = z
            props['C'] = chindex

            # count the number of objects
            values['Number'] = props.shape[0]
            # values['Number'] = len(regions) - 1
            print('Objects found: ', values['Number'])

            # update dataframe containing the number of objects
            objects = objects.append(pd.DataFrame(values, index=[0]), ignore_index=True)

            results = results.append(props, ignore_index=True)

            image_counter += 1
            # optional display of results
            if image_counter - 1 in show_image:
                plot_results(image2d, mask, props, add_bbox=True)

# reorder dataframe with single objects
new_order = list(results.columns[-4:]) + list(results.columns[:-4])
results = results.reindex(columns=new_order)


img.close()

print('Done')

print(objects)
# print(results)
