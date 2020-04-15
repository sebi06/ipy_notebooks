# -*- coding: utf-8 -*-

#################################################################
# File        : test_nuclei_segmentation.py
# Version     : 0.3
# Author      : czsrh
# Date        : 11.04.2020
# Institution : Carl Zeiss Microscopy GmbH
#
# Copyright (c) 2020 Carl Zeiss AG, Germany. All Rights Reserved.
#################################################################


use_method = 'scikit'
# use_method = 'cellpose'
# use_method = 'zentf'


import sys
import time
import os
from glob import glob
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import imgfileutils as imf
from scipy import ndimage
from aicsimageio import AICSImage, imread
from skimage import exposure
from skimage.morphology import watershed, dilation
from skimage.feature import peak_local_max
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import random_walker
from skimage import io, measure, segmentation
from skimage.filters import threshold_otsu, threshold_triangle, rank
from skimage.segmentation import clear_border
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.util import invert
from skimage.filters import median, gaussian
from skimage.morphology import closing, square
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.morphology import disk, square, ball
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


def apply_watershed(binary, min_distance=10):

    # create distance map
    distance = ndimage.distance_transform_edt(binary)

    # determine local maxima
    local_maxi = peak_local_max(distance,
                                min_distance=min_distance,
                                indices=False,
                                labels=binary)

    # label maxima
    markers, num_features = ndimage.label(local_maxi)

    # apply watershed
    mask = watershed(-distance, markers,
                     mask=binary,
                     watershed_line=True).astype(np.int)

    return mask


def apply_watershed_adv(image2d,
                        segmented,
                        filtermethod_ws='median',
                        filtersize_ws=3,
                        min_distance=2,
                        radius=1):

    # convert to float
    image2d = image2d.astype(np.float)

    # rescale 0-1
    image2d = rescale_intensity(image2d, in_range='image', out_range=(0, 1))

    # filter image
    if filtermethod == 'median':
        image2d = median(image2d, selem=disk(filtersize_ws))
    if filtermethod == 'gauss':
        image2d = gaussian(image2d, sigma=filtersize_ws, mode='reflect')

    # create the seeds
    peaks = peak_local_max(image2d,
                           labels=label(segmented),
                           min_distance=min_distance,
                           indices=False)

    # create the seeds
    seed = dilation(peaks, selem=disk(radius))

    # create watershed map
    watershed_map = -1 * distance_transform_edt(segmented)

    # create mask
    mask = watershed(watershed_map,
                     markers=label(seed),
                     mask=segmented,
                     watershed_line=True).astype(np.int)

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


def segment_threshold(image2d,
                      filtermethod='median',
                      filtersize=3,
                      threshold='triangle',
                      split_ws=True,
                      min_distance=30,
                      ws_method='ws_adv',
                      radius=1):

    # filter image
    if filtermethod == 'none':
        image2d_filtered = image2d
    if filtermethod == 'median':
        image2d_filtered = median(image2d, selem=disk(filtersize))
    if filtermethod == 'gauss':
        image2d_filtered = gaussian(image2d, sigma=filtersize, mode='reflect')

    # threshold image and run marker-based watershed
    binary = autoThresholding(image2d_filtered, method=threshold)

    # apply watershed
    if split_ws:

        if ws_method == 'ws':
            mask = apply_watershed(binary,
                                   min_distance=min_distance)

        if ws_method == 'ws_adv':
            mask = apply_watershed_adv(image2d, binary,
                                       min_distance=min_distance,
                                       radius=radius)

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
        ax[0] = add_boundingbox(props, ax[0])

    plt.show()


def add_boundingbox(props, ax2plot):
    """Add bounding boxes for objects to the current axes

    Arguments:
        props {Pandas DataFrame} -- DataFrame contained the measured parameters
                                    for the bounding boxes
        ax2plot {MatplotLib axes} -- The Axes contained the images where
                                     the boxes should be drawn

    Returns:
        [MatplotLib axes] -- The axes inclusing the bounding boxes.
    """
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

    return ax2plot


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


def showheatmap(heatmap, parameter2display,
                fontsize_title=12,
                fontsize_label=10,
                colormap='Blues',
                linecolor='black',
                linewidth=1.0,
                save=False,
                savename='Heatmap.png',
                robust=True,
                filename='Test.czi',
                dpi=100):

    # create figure with subplots
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # create the heatmap
    ax = sns.heatmap(heatmap,
                     ax=ax,
                     cmap=colormap,
                     linecolor=linecolor,
                     linewidths=linewidth,
                     square=True,
                     robust=robust,
                     annot=False,
                     cbar_kws={"shrink": 0.68})

    # customize the plot to your needs
    ax.set_title(parameter2display,
                 fontsize=fontsize_title,
                 fontweight='normal')

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize_label)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize_label)

    # modify the labels of the colorbar
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=fontsize_label)

    if save:
        savename = filename[:-4] + '_HM_' + parameter2display + '.png'
        fig.savefig(savename,
                    dpi=dpi,
                    orientation='portrait',
                    transparent=False,
                    frameon=False)
        print('Heatmap image saved as: ', savename)
    else:
        savename = False

    return savename


def getrowandcolumn(platetype=96):
    """
    :param platetype - number total wells of plate (6, 24, 96, 384 or 1536)
    :return nr - number of rows of wellplate
    :return nc - number of columns for wellplate
    """
    platetype = int(platetype)

    if platetype == 6:
        nr = 2
        nc = 3
    elif platetype == 24:
        nr = 4
        nc = 6
    elif platetype == 96:
        nr = 8
        nc = 12
    elif platetype == 384:
        nr = 16
        nc = 24
    elif platetype == 1536:
        nr = 32
        nc = 48

    return nr, nc


def create_heatmap(platetype=96):

    # create heatmap based on the platetype
    nr, nc = getrowandcolumn(platetype=platetype)
    heatmap_array = np.full([nr, nc], np.nan)

    return heatmap_array


def convert_array_to_heatmap(hmarray, nr, nc):
    """Get the labels for a well plate and create a data frame from the numpy array

    Arguments:
        hmarray {ndarray} -- The numpy array containing the actual heatmap.
        nr {integer} -- number of rows for the well plate
        nc {integer} -- number of colums for the wellplate

    Returns:
        [Pandas DataFrame] -- A Pandas dataframe with the respective
                              row and columns labels
    """

    lx, ly = extract_labels(nr, nc)
    heatmap_dataframe = pd.DataFrame(hmarray, index=ly, columns=lx)

    return heatmap_dataframe


def extract_labels(nr, nc):
    """
    Define helper function to be able to extract the well labels depending
    on the actual wellplate type. Currently supports 96, 384 and 1536 well plates.

    :param nr: number of rows of the wellplate, e.g. 8 (A-H) for a 96 wellplate
    :param nc: number of columns of the wellplate, e.g. 12 (1-12) for a 96 wellplate
    :return: lx, ly are list containing the actual row and columns IDs
    """

    # labeling schemes
    labelX = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
              '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
              '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36',
              '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', ]

    labelY = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
              'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF']

    lx = labelX[0:nc]
    ly = labelY[0:nr]

    return lx, ly

###############################################################################


filenames = [r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate/B4_B5_S=8_4Pos_perWell_T=2_Z=1_CH=1.czi',
             r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate/96well-SingleFile-Scene-05-A5-A5.czi',
             r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate/testwell96.czi',
             r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\testwell96.czi',
             r'C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Castor\testwell96 - A1_1024x1024_0.czi',
             r'segment_nuclei_CNN.czi']

filename = filenames[2]

# define platetype and get number of rows and columns
platetype = 96
nr, nc = getrowandcolumn(platetype=platetype)

# get AICSImageIO object using the python wrapper for libCZI
img = AICSImage(filename)
SizeS = img.size_s
SizeT = img.size_t
SizeZ = img.size_z

chindex = 0  # channel containing the objects, e.g. the nuclei
minsize = 200  # minimum object size [pixel]
maxsize = 5000  # maximum object size [pixel]

# define cutout size for subimage
cutimage = False
startx = 0
starty = 0
width = 600
height = 600

# define columns names for dataframe
cols = ['S', 'T', 'Z', 'C', 'Number']
objects = pd.DataFrame(columns=cols)

# optional dipslay of "some" results - empty list = no display
show_image = [0]

# for testing
#SizeS = 1

# threshold parameters
filtermethod = 'median'
#filtermethod = None
filtersize = 3
threshold = 'triangle'

# use watershed for splitting
use_ws = True
ws_method = 'ws_adv'
filtermethod_ws = 'median'
filtersize_ws = 3
min_distance = 15
radius_dilation = 1

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

    # define tile overlap factor for MightyMosaic
    overlapfactor = 1

    # Load the model
    MODEL_PATH = 'model_folder'
    model = tf.keras.models.load_model(MODEL_PATH)

    # Determine input shape required by the model and crop input image
    tile_height, tile_width = model.signatures["serving_default"].inputs[0].shape[1:3]
    print('ZEN TF Model Tile Dimension : ', width, height)

###########################################################################

image_counter = 0
results = pd.DataFrame()

# get the CZI metadata
# get the metadata from the czi file
md = imf.get_metadata_czi(filename, dim2none=False)


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
                mask = segment_threshold(image2d,
                                         filtermethod=filtermethod,
                                         filtersize=filtersize,
                                         threshold=threshold,
                                         split_ws=use_ws,
                                         min_distance=min_distance,
                                         ws_method=ws_method,
                                         radius=radius_dilation)

            if use_method == 'zentf':

                classlabel = 1

                # check if tiling is required
                if image2d.shape[0] > tile_height or image2d.shape[1] > tile_width:
                    binary = segment_zentf_tiling(image2d, model,
                                                  tilesize=tile_height,
                                                  classlabel=classlabel,
                                                  overlap_factor=overlapfactor)

                elif image2d.shape[0] == tile_height and image2d.shape[1] == tile_width:
                    binary = segment_zentf(image2d, model,
                                           classlabel=classlabel)

                elif image2d.shape[0] < tile_height or image2d.shape[1] < tile_width:

                    # do padding
                    image2d_padded, pad = add_padding(image2d,
                                                      input_height=tile_height,
                                                      input_width=tile_width)

                    # run prediction on padded image
                    binary_padded = segment_zentf(image2d_padded, model,
                                                  classlabel=classlabel)

                    # remove padding from result
                    binary = binary_padded[pad[0]:tile_height - pad[1], pad[2]:tile_width - pad[3]]

                # apply watershed
                if use_ws:
                    if ws_method == 'ws':
                        mask = apply_watershed(binary,
                                               min_distance=min_distance)
                    if ws_method == 'ws_adv':
                        mask = apply_watershed_adv(image2d, binary,
                                                   min_distance=min_distance,
                                                   radius=radius_dilation)
                if not use_ws:
                    # label the objects
                    mask, num_features = ndimage.label(binary)
                    mask = mask.astype(np.int)

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

            # add wellinformation for CZI metadata
            props['WellId'] = md['Well_ArrayNames'][s]
            props['Well_ColId'] = md['Well_ColId'][s]
            props['Well_RowId'] = md['Well_RowId'][s]

            # add plane indicies
            props['S'] = s
            props['T'] = t
            props['Z'] = z
            props['C'] = chindex

            # count the number of objects
            values['Number'] = props.shape[0]
            # values['Number'] = len(regions) - 1
            print('Objects found: ', values['Number'])

            # update dataframe containing the number of objects
            objects = objects.append(pd.DataFrame(values, index=[0]),
                                     ignore_index=True)

            results = results.append(props, ignore_index=True)

            image_counter += 1
            # optional display of results
            if image_counter - 1 in show_image:
                plot_results(image2d, mask, props, add_bbox=True)

# reorder dataframe with single objects
new_order = list(results.columns[-7:]) + list(results.columns[:-7])
results = results.reindex(columns=new_order)

img.close()

print('Done')

# create heatmap array with NaNs
heatmap_numobj = create_heatmap(platetype=platetype)
heatmap_param = create_heatmap(platetype=platetype)

for well in md['WellCounter']:
    # extract all entries for specific well
    well_results = results.loc[results['WellId'] == well]

    # get the descriptive statistics for specific well
    stats = well_results.describe(include='all')

    # get the column an row indices for specific well
    col = np.int(stats['Well_ColId']['mean'])
    row = np.int(stats['Well_RowId']['mean'])

    # add value for number of objects to heatmap_numobj
    heatmap_numobj[row - 1, col - 1] = stats['WellId']['count']

    # add value for specifics params to heatmap
    heatmap_param[row - 1, col - 1] = stats['area']['mean']

df_numobjects = convert_array_to_heatmap(heatmap_numobj, nr, nc)
df_params = convert_array_to_heatmap(heatmap_param, nr, nc)

# show a heatmap

# define parameter to display a single heatmap
parameter2display = 'ObjectNumbers'
colormap = 'YlGnBu'

# show the heatmap for a single parameter
savename_single = showheatmap(df_numobjects, parameter2display,
                              fontsize_title=16,
                              fontsize_label=16,
                              colormap=colormap,
                              linecolor='black',
                              linewidth=3.0,
                              save=False,
                              filename=filename,
                              dpi=300)

# print(objects)
# print(results[:5])

plt.show()
