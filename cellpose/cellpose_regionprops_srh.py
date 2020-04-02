
import sys
import time
import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
# import imgfileutils as imf
from scipy import ndimage

import mxnet
from cellpose import plot, transforms
from cellpose import models, utils
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


def segment_nuclei_scikit(image2d,
                          filtermethod='median',
                          filtersize=3,
                          threshold='triangle',
                          watershed_fp=13):

    # filter image
    if filtermethod == 'median':
        image2d = median(image2d, selem=disk(filtersize))
    if filtermethod == 'gauss':
        image2d = gaussian(image2d, sigma=filtersize, mode='reflect')

    # threshold image and run marker-based watershed
    binary = autoThresholding(image2d, method='triangle')
    distance = ndimage.distance_transform_edt(binary)
    local_maxi = peak_local_max(distance,
                                # min_distance=3,
                                indices=False,
                                footprint=np.ones((watershed_fp, watershed_fp)),
                                labels=binary)

    markers, num_features = ndimage.label(local_maxi)
    mask = watershed(-distance, markers, mask=binary, watershed_line=True)

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


###############################################################################

# filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate/testwell96.czi'
filename = r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\testwell96.czi'

# get AICSImageIO object using the python wrapper for libCZI
img = AICSImage(filename)
SizeS = img.size_s
SizeT = img.size_t
SizeZ = img.size_z

chindex = 0  # channel containing the nuclei
minsize = 200  # minimum object size
maxsize = 3000  # maximum object size

# define columns names for dataframe
cols = ['S', 'T', 'Z', 'C', 'Number']
objects = pd.DataFrame(columns=cols)

# optional dipslay of "some" results - empty list = no display
show_image = [0]

# for testing
SizeS = 1

#use_method = 'scikit'
use_method = 'cellpose'

# load the ML model from cellpose when needed
if use_method == 'cellpose':
    # load cellpose model for cell nuclei using GPU or CPU
    print('Loading Cellpose Model ...')
    model = models.Cellpose(device=set_device(), model_type='nuclei')

    # define list of channels for cellpose
    # channels = SizeS * SizeT * SizeZ * [0, 0]
    channels = [0, 0]

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
            image2d = image2d[0:1000, 0:1000]

            if use_method == 'cellpose':
                # get the mask for the current image
                mask = segment_nuclei_cellpose(image2d, model,
                                               rescale=None,
                                               channels=channels)

            if use_method == 'scikit':
                mask = segment_nuclei_scikit(image2d,
                                             filtermethod='median',
                                             filtersize=3,
                                             threshold='triangle',
                                             watershed_fp=25)

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
            #props = [r for r in props if r.area >= minsize]

            props['S'] = s
            props['T'] = t
            props['Z'] = z
            props['C'] = chindex

            # count the number of objects
            values['Number'] = props.shape[0]
            #values['Number'] = len(regions) - 1
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
print(results)
