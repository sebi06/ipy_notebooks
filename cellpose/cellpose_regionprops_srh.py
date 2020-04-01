import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches

from scipy import ndimage

import numpy as np
import time
import os
import mxnet
import sys
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

# check if GPU working, and if so use it
use_gpu = utils.use_gpu()

print('Use GPU: ', use_gpu)

if use_gpu:
    device = mxnet.gpu()
else:
    device = mxnet.cpu()

# load cellpose model for cell nuclei using GPU or CPU
model = models.Cellpose(device=device, model_type='nuclei')


def segment_nuclei(image2d, model,
                   channels=[0, 0],
                   rescale=None):

    # get the mask for a single image
    masks, _, _, _ = model.eval([image2d], rescale=rescale, channels=channels)

    return masks[0]


def get_regions(mask,
                inside_only=True,
                minsize=10):

    if inside_only:
        mask = segmentation.clear_border(mask)

    # get the regions from the mask
    regions = regionprops(mask)

    # filter by size
    regions = [r for r in regions if r.area >= minsize]

    return regions


###############################################################################
# filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate/testwell96.czi'
filename = r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\testwell96.czi'


# get AICSImageIO object using the python wrapper for libCZI
img = AICSImage(filename)
SizeS = img.size_s
SizeT = img.size_t
SizeZ = img.size_z

# define list of channels for cellpose
channels = SizeS * SizeT * SizeZ * [0, 0]
channels = [0, 0]
chindex = 0
cols = ['S', 'T', 'Z', 'C', 'Number']

# create dataframe for number of objects
cells = pd.DataFrame(columns=cols)

minsize = 50

# for testing
#SizeS = 3

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
            #image2d = image2d[600:900, 400:700]

            # get the mask for the current image
            mask = segment_nuclei(image2d, model,
                                  rescale=None,
                                  channels=channels)

            # get the regions from the mask, clear border and filter by size
            regions = get_regions(mask,
                                  inside_only=True,
                                  minsize=minsize)

            # count the number of objects
            values['Number'] = len(regions) - 1
            print('Objects found: ', values['Number'])
            print('values : ', values)

            # update dataframe
            cells = cells.append(pd.DataFrame(values, index=[0]), ignore_index=True)

img.close()

print('Done')

print(cells)

"""
# Get an AICSImage object
img = AICSImage(filename)
data = img.get_image_data("YX", S=0, T=0, Z=0, C=0)
# data = data[600:900, 400:700]
data = data[200:1000, 200:1000]
img.close()


model = models.Cellpose(device=mxnet.gpu(), model_type='nuclei')

# files_raw = sorted(glob('*/*/*.tif'))
# files = list(filter(lambda f: f.startswith('wt') or f.startswith('mut'), files_raw))
# images = map(io.imread, files)

channels = [0, 0]

scale = 1.0  # µm per pixel
# header = True

files = [filename]
images = [data]

for filename, image in zip(files, images):
    print(f'{filename} started')

    # get cell mask
    masks, _, _, _ = model.eval([image], rescale=None, channels=channels)
    mask = segmentation.clear_border(masks[0])

    image_label_overlay = label2rgb(mask, image=image, bg_label=0)

    current_regions = regionprops(mask)

    # make and save dataframe
    props = pd.DataFrame(
        measure.regionprops_table(
            mask, properties=('label', 'area', 'centroid')
        )
    ).set_index('label')

    props.loc[props['area'] >= 100]
    # regions = [r for r in regions if r.area > 100]

    props['filename'] = filename
    props['type'] = 'test'
    props['area (µm²)'] = props['area'] * (scale**2)
    props.to_csv('out.csv', mode='a', header=header)
    header = False

    # make and save figure
    marked = segmentation.mark_boundaries(image, mask, color=(0, 0, 0), mode='thick')
    dpi = 100
    figsize = (image.shape[1] / dpi, image.shape[0] / dpi)
    # fig = plt.Figure(figsize=figsize, dpi=dpi)
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # ax = fig.add_axes([0, 0, 1, 1])
    # ax.imshow(marked, interpolation='nearest', vmin=marked.min(), vmax=marked.max())
    marked = exposure.rescale_intensity(marked, in_range='image', out_range=(0, 1))

    # ax[0].imshow(marked, interpolation='nearest', clim=[marked.min(), marked.max()])
    ax[0].imshow(image,
                 cmap=plt.cm.gray,
                 interpolation='nearest',
                 clim=[image.min(), image.max() * 0.5])

    ax[1].imshow(image_label_overlay,
                 clim=[image.min(), image.max() * 0.5])

    ax[0].set_title('Original', fontsize=12)
    ax[1].set_title('Masks', fontsize=12)


    for label, (area, y, x, fn, t, aa) in props.iterrows():

        ax[0].text(x, y, str(int(label)),
                   verticalalignment='center',
                   horizontalalignment='center')

        ax[1].text(x, y, str(int(label)),
                   verticalalignment='center',
                   horizontalalignment='center')

    for region in regionprops(mask):
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False,
                                      edgecolor='red',
                                      linewidth=2)
            ax[0].add_patch(rect)

    plt.show()
    output_filename = filename[:-4] + '_segmentation.png'
    # fig.savefig(output_filename)

    print(f'{filename} done')
"""
