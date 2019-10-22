#################################################################
# File       : particleanalyzer.py
# Version    : 0.5
# Author     : czsrh
# Date       : 22.10.2019
# Insitution : Carl Zeiss Microscopy GmbH
#
#
# Copyright (c) 2018 Carl Zeiss AG, Germany. All Rights Reserved.
#################################################################

import sys
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk
from skimage.external import tifffile
import scipy.ndimage as nd
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import czifile as zis
import xmltodict
import imgfileutils as imf


def classify(value, particleclasses):
    """
    The particle class is expected to be a dictioary in this form

    particleclasses={'Classletter',: [MinimumSize, MaximumSize]}

    Example:

    particleclasses = {'B': [5, 15],
                       'C': [15, 25],
                       'D': [25, 50],
                       'E': [50, 1000],
                       'F': [100, 150],
                       'G': [150, 200],
                       'H': [200, 400],
                       'I': [400, 600],
                       'J': [600, 1000],
                       'K': [1000, 1500],
                       'L': [1500, 2000],
                       'M': [2000, 3000],
                       'N': [3000, np.inf]
                       }
    """
    # get list of keys
    classletterlist = list(particleclasses)

    # loop through all classes
    for cl in classletterlist:

        # get the minimum and maximum values for the current class
        minvalue = particleclasses[cl][0]
        maxvalue = particleclasses[cl][1]

        # check if the value is inside the interval and assign letter to it
        if minvalue <= value < maxvalue:
            sizeclass = cl

    return sizeclass


def autoThresholding(image2d,
                     method='global_otsu',
                     radius=10,
                     value=50):

    # calculate global Otsu threshold
    if method == 'global_otsu':
        thresh = threshold_otsu(image2d)
        binary = image2d > thresh

    # calculate local Otsu threshold
    if method == 'local_otsu':
        thresh = rank.otsu(image2d, disk(radius))
        binary = image2d > thresh

    if method == 'value_based':
        binary = image2d >= value

    return binary


def findhistogrammpeak(values, bins):

    v = np.where(values == values.max())
    most_frequent_value = np.round(bins[v], 0)

    return most_frequent_value


"""
def readczi(filename, replacezero=True):

    czi = zis.CziFile(filename)
    array = czi.asarray()
    md = czi.metadata()

    metadata = xmltodict.parse(md)

    czimd = {}

    czimd['Axes'] = czi.axes
    czimd['Shape'] = czi.shape

    try:
        czimd['Experiment'] = metadata['ImageDocument']['Metadata']['Experiment']
    except:
        czimd['Experiment'] = None

    try:
        czimd['HardwareSetting'] = metadata['ImageDocument']['Metadata']['HardwareSetting']
    except:
        czimd['HardwareSetting'] = None

    try:
        czimd['CustomAttributes'] = metadata['ImageDocument']['Metadata']['CustomAttributes']
    except:
        czimd['CustomAttributes'] = None

    czimd['Information'] = metadata['ImageDocument']['Metadata']['Information']
    czimd['PixelType'] = czimd['Information']['Image']['PixelType']
    czimd['SizeX'] = czimd['Information']['Image']['SizeX']
    czimd['SizeY'] = czimd['Information']['Image']['SizeY']

    try:
        czimd['SizeZ'] = czimd['Information']['Image']['SizeZ']
    except:
        czimd['SizeZ'] = None

    try:
        czimd['SizeC'] = czimd['Information']['Image']['SizeC']
    except:
        czimd['SizeC'] = None

    try:
        czimd['SizeT'] = czimd['Information']['Image']['SizeT']
    except:
        czimd['SizeT'] = None

    try:
        czimd['SizeM'] = czimd['Information']['Image']['SizeM']
    except:
        czimd['SizeM'] = None

    try:
        czimd['Scaling'] = metadata['ImageDocument']['Metadata']['Scaling']
        czimd['ScaleX'] = float(czimd['Scaling']['Items']['Distance'][0]['Value']) * 1000000
        czimd['ScaleY'] = float(czimd['Scaling']['Items']['Distance'][1]['Value']) * 1000000
        try:
            czimd['ScaleZ'] = float(czimd['Scaling']['Items']['Distance'][2]['Value']) * 1000000
        except:
            czimd['ScaleZ'] = None
    except:
        czimd['Scaling'] = None

    try:
        czimd['DisplaySetting'] = metadata['ImageDocument']['Metadata']['DisplaySetting']
    except:
        czimd['DisplaySetting'] = None

    try:
        czimd['Layers'] = metadata['ImageDocument']['Metadata']['Layers']
    except:
        czimd['Layers'] = None

    if replacezero:
        array = replaceZeroNaN(array, value=0)

    return array, czimd


def replaceZeroNaN(data, value=0):

    data = data.astype('float')
    data[data == value] = np.nan

    return data
"""


def getmaxinscribedcircle(labelimage):
    """
    Code taken from:

    https://stackoverflow.com/questions/38598690/how-to-find-the-diameter-of-objects-using-image-processing-in-python/38616904

    This gives the largest inscribed circle (or sphere in 3D). The find_objects function is quite handy.
    It returns a list of Python slice objects, which you can use to index into the image at the specific
    locations containing the blobs. These slices can of course be used to index into the distance transform image.
    Thus the largest value of the distance transform inside the slice is the radius you're looking for.

    There is one potential gothcha of the above code: the slice is a square (or cubic) section
    so might contain small pieces of other blobs if they are close together.
    You can get around this with a bit more complicated logic as follows:

    radii = [np.amax(dt[slices[i]]*(labels[slices[i]] == (i+1))) for i in range(nlabels)]

    The above version of the list comprehension masks the distance transform with the blob that is supposed to be indexed by the slice, thereby removing any unwanted interference from neighboring blobs.

    """

    # clacluate the distance transform
    dt = nd.distance_transform_edt(labelimage)
    # find objects in image
    slices = nd.find_objects(input=labelimage)
    # find the maximum inscribed circle
    mic_radius = [np.amax(dt[s]) for s in slices]

    return mic_radius


def apply_watershed(bool_label_image, footprint=10):

    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(bool_label_image)

    # calculate the local maxima
    local_maxi = peak_local_max(distance,
                                indices=False,
                                footprint=np.ones((footprint, footprint)),
                                labels=bool_label_image)

    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=bool_label_image)

    return labels


def create_rectangles(ax, properties, region, offset=10, lw=3):

    # draw rectangle around segmented particles and make the bounding rectangle bigger
    minr, minc, maxr, maxc = region.bbox
    minr = minr - offset
    maxr = maxr + offset
    minc = minc - offset
    maxc = maxc + offset

    # add rectangle and legend, but only if "this" legend was not already added
    if properties['IsFiber']:
        ls = 'dotted'
    if not properties['IsFiber']:
        ls = 'solid'
    if properties['IsFiber'] == 'ND':
        ls = 'dashed'

    # draw rectangles depending on class
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False,
                              edgecolor=properties['ClassColor'],
                              linewidth=lw,
                              linestyle=ls,
                              label=str(properties['Class']) if str(properties['Class']) not in ax.get_legend_handles_labels()[1] else '')

    ax.add_patch(rect)
    ax.legend(loc='upper left')

    return ax, rect


def count_objecttypes(objects):

    num_all_objects = len(objects)
    num_particles = 0
    num_fibers = 0
    num_ND = 0

    for obj in objects.keys():

        fiber = objects[obj]['IsFiber']
        if fiber:
            num_fibers = num_fibers + 1
        if not fiber:
            num_particles = num_particles + 1
        if fiber == 'ND':
            num_ND = num_ND + 1

    return num_all_objects, num_fibers, num_particles, num_ND


def calc_grid(num_all_objects):

    # calculate number of needed columns and rows for particle grid plot
    nc = np.int(np.round(np.sqrt(num_all_objects)))
    nr = nc + 1
    nrr = np.int(np.round(nr * nc / 10, 0) + 2)

    return nrr, nc


def get_particleproperties(region, img_label, image, particlesizeclasses, particlesizecolors, scaleXY=0.1):

    NUM_COLORS = len(list(particlesizeclasses.keys()))

    # create distinct color for every class for later visualization
    jet = plt.get_cmap('jet')
    COLORS_RECT = jet(np.linspace(0, 2, NUM_COLORS))

    # create dictionary for the individual particle properties
    properties = {}

    # calculate FeretMax
    feretmax = region.major_axis_length * scaleXY
    feretmin = region.minor_axis_length * scaleXY

    # calculate perimeter, area and fiberlength of object
    perimeter = region.perimeter * scaleXY
    area = region.area * scaleXY**2

    fiberlength = 0.25 * (area + np.sqrt(perimeter**2 - 16 * area))
    # check if it was possible to calculate the fiberlength
    if np.isnan(fiberlength):
        fiberlength = 0.0

    # classify objects according the th given size class definitions
    properties['Class'] = classify(feretmax, particlesizeclasses)

    # assign color deopending on the size class
    properties['ClassColor'] = COLORS_RECT[particlesizecolors[properties['Class']], :]

    # fill the dictionary
    properties['Perimeter'] = perimeter
    properties['Area'] = area
    properties['FiberLength'] = fiberlength
    properties['FeretMax'] = feretmax
    properties['FeretMin'] = feretmin
    # convert boolean array to 0-1 array
    properties['LabelImage'] = region.image * 1

    # calculate maximum inscribed circle diameter
    mic_radius = float(getmaxinscribedcircle(properties['LabelImage'])[0])
    properties['maxMIC'] = mic_radius * 2 * scaleXY

    # calculate ratio length - width
    if feretmin > 0:
        # calculate the ratio from feretmax and feretmin
        properties['FeretRatio'] = feretmax / feretmin

        # check if this is a fiber using a special criterion
        if properties['maxMIC'] <= 50 and (properties['FiberLength'] / properties['maxMIC']) > 20:
            properties['IsFiber'] = True
        else:
            properties['IsFiber'] = False

    elif feretmin == 0:
        # on case feretmin = 0 (should never be) set values
        properties['FeretRatio'] = np.NaN
        #print('FeretRatio set to NaN')
        properties['IsFiber'] = 'ND'

    # get the intensity image
    ri = region.intensity_image
    ri = ri.astype('float')
    ri[ri == 0] = np.NaN
    properties['IntensityImage'] = ri

    # store parameters of current object in dictionary for all objects

    return properties


def find_fibers(objects):

    # find all fibers in object list
    fiber_ids = []
    for k, v in objects.items():
        if objects[k]['IsFiber']:
            if objects[k]['FiberLength'] > 0.0:
                fiber_ids.append(k)

    return fiber_ids
