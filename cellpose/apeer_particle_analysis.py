#################################################################
# File       : apeer_particle_analysis.py
# Version    : 0.7
# Author     : czsrh
# Date       : 14.01.2020
# Institution : Carl Zeiss Microscopy GmbH
#
#
# Copyright (c) 2018 Carl Zeiss AG, Germany. All Rights Reserved.
#################################################################

import imgfileutils as imf
import seaborn as sns
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.color import label2rgb
from skimage.morphology import closing, square
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops
from skimage import img_as_ubyte
from skimage.util import invert
from skimage import exposure
from skimage.util.dtype import dtype_range
from skimage.external import tifffile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools as it
import particleanalyzer as pa
from pathlib import Path
import sys
import os
from apeer_ometiff_library import io, processing, omexmlClass
import warnings
warnings.filterwarnings('ignore')


def analyze_particles(filename_image,
                      th_rel=0.65,
                      remove_small=False,
                      min_particle_area_micron=1.0,
                      mv_feretmax=5.0,
                      pa_parameter='FeretMax',
                      separator=';',
                      figdpi=300):
    """
    if filename_image.lower().endswith('.ome.tiff') or filename_image.lower().endswith('.ome.tif'):

        # Return value is an array of order (T, Z, C, X, Y)
        print('Getting OME-TIFF as array ...')
        (array, omexml) = io.read_ometiff(filename_image)
        metadata = imf.get_metadata(filename_image, series=0)

    if filename_image.lower().endswith('.czi'):

        # get only the metadata
        #metadata = imf.get_metadata(filename)
        # get the array and the metadata
        print('Getting CZI as array ...')
        array, metadata = imf.get_array_czi(filename_image, replacezero=False)
    """

    # read metadata and array differently for OME-TIFF or CZI data
    if filename_image.lower().endswith('.ome.tiff') or filename_image.lower().endswith('.ome.tif'):

        # Return value is an array of order (T, Z, C, X, Y)
        print('Getting OME-TIFF as array ...')
        (array, omexml) = io.read_ometiff(filename_image)
        metadata, add_metadata = imf.get_metadata(filename_image, series=0)

    if filename_image.lower().endswith('.czi'):

        # get the array and the metadata
        array, metadata, add_metadata = imf.get_array_czi(filename_image, return_addmd=False)

    # for later use reduce the filename to its basename
    filename = os.path.basename(filename_image)

    # print the important metadata parameters
    print(metadata['Axes'])
    print(metadata['Shape'])
    print(metadata['SizeT'])
    print(metadata['SizeZ'])
    print(metadata['SizeC'])
    print(metadata['SizeX'])
    print(metadata['SizeY'])
    print(metadata['XScale'])
    print(metadata['YScale'])
    print(metadata['ZScale'])

    # squeeze array to 2D image and calculate min and max pixel value
    array = np.squeeze(array)
    minpix = array.min()
    maxpix = array.max()

    # define number of bins for histogram and create figure
    bins = 256
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # display original image subset
    ax[0].imshow(array, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title('Original Image', fontsize=16)
    #xmin, xmax = dtype_range[array.dtype.type]

    # calculate histogram and return values
    values, bins, bars = ax[1].hist(array.ravel(), bins=bins)

    # find the index for the histogram peak
    v = np.where(values == values.max())

    # get the most frequent pixel value
    most_frequent_value = pa.findhistogrammpeak(values, bins)
    luminosity = np.round(most_frequent_value / 255 * 100, 0)

    # calculate threshold pixel value from relative threshold
    threshold = np.round(most_frequent_value * th_rel, 0)

    # customize plots
    ax[1].set_title('Histogram', fontsize=16)
    ax[1].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax[1].set_xlabel('Pixel intensity', fontsize=16)
    #ax[1].set_xlim(xmin, xmax)

    # display distribution
    axh = ax[1].twinx()
    axh.vlines(most_frequent_value, 0, 1, colors='r',
               linestyles='solid',
               lw=5,
               label='most frequent pixel value')

    axh.vlines(threshold, 0, 1, colors='g',
               linestyles='solid',
               lw=5,
               label='pixel threshold')
    axh.set_ylim(0, 1)
    axh.legend(loc='upper left')

    # turn off tick labels
    axh.set_yticklabels([])

    # save output figure
    savename_histogram_image = filename[:-4] + '_Histogram.png'
    fig.savefig(savename_histogram_image, dpi=figdpi, orientation='portrait', transparent=False, frameon=False)

    print('Index for this Value      : ', v[0])
    print('Most frequent Pixel Value : ', most_frequent_value)
    print('Luminosity [%]            : ', luminosity)
    print('Relative Threshold Value  : ', th_rel)
    print('abs. Threshold Value      : ', threshold)

    # apply the threshold to the image and invert the result to get the dark particles
    th_img = pa.autoThresholding(array, method='value_based', value=threshold)
    th_img = invert(th_img)

    # dosplay the result
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    ax[0].imshow(array, cmap=plt.cm.gray, interpolation='nearest')
    ax[1].imshow(th_img)

    ax[0].set_title('Original Image', fontsize=16)
    ax[1].set_title('Thresholded Image', fontsize=16)

    # save threshold image
    savename_threshold_image = filename[:-4] + '_Threshold.png'
    fig.savefig(savename_threshold_image, dpi=figdpi, orientation='portrait', transparent=False, frameon=False)
    plt.close(fig)

    # remove particles from the edge
    th_img = clear_border(th_img)

    # label the particles
    img_label, num_label = label(th_img, background=0, return_num=True, connectivity=2)
    print('Initial Number of Particles       : ', num_label)

    # remove small objects
    if remove_small:

        # min_particle_size_pixel = np.int(np.round(min_particle_area_micron / (metainfo['XScale']**2), 0))
        min_particle_size_pixel = np.int(np.round(min_particle_area_micron / (metadata['XScale']**2), 0))

        # if small object will be removed make sure the size is at least = 1 pixel
        if min_particle_size_pixel == 0.0:
            min_particle_size_pixel = 1.0

        print('Minimum Particle Size [micron**2] :  {:.2f}'.format(min_particle_area_micron))
        print('Minimum Particle Size [pixel]     :  {:.2f}'.format(min_particle_size_pixel))

        img_label = remove_small_objects(img_label, min_particle_size_pixel, in_place=False)

    # define particle classes and the respective colors
    pclasses = {'A': [0, 5],
                'B': [5, 15],
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

    pcolors = {'A': 0,
               'B': 1,
               'C': 2,
               'D': 3,
               'E': 4,
               'F': 5,
               'G': 6,
               'H': 7,
               'I': 8,
               'J': 9,
               'K': 10,
               'L': 11,
               'M': 12,
               'N': 13
               }

    # show original image only and labels (optional)
    #image_label_overlay = label2rgb(img_label, image=array, bg_label=0)
    #image_label_overlay = img_label
    image_label_overlay = array

    # display the labelled image
    fig, ax_label = plt.subplots(figsize=(16, 16))
    ax_label.imshow(image_label_overlay, cmap=plt.cm.gray, interpolation='nearest')

    # minimum value for parameter to disgard a particle
    #mv_feretmax = 5

    # offset to make bounding boxes a "bit bigger"
    offset = 10
    drawrect = True

    # create emtpy dictionary for all the particles and lust for particle classes
    objects = {}
    #particleclasses = []

    skipped = 0

    # loop over all detected objects
    # for region in regionprops(img_label, intensity_image=array, cache=True, coordinates='xy'):
    for region in regionprops(img_label, intensity_image=array, cache=True):

        properties = pa.get_particleproperties(region, img_label, array, pclasses, pcolors, scaleXY=metadata['XScale'])

        # store parameters of current object in dictionary for all objects
        objects[region.label] = properties

        if drawrect and objects[region.label]['FeretMax'] >= mv_feretmax:
            # draw rectangle around segmented particles and make the bounding rectangle bigger
            ax, rect = pa.create_rectangles(ax_label, properties, region, offset=offset, lw=3)

        # check for small particles and delete entry from dict
        if objects[region.label]['FeretMax'] < mv_feretmax or np.math.isnan(objects[region.label]['FeretRatio']) == True:
            objects.pop(region.label, None)
            skipped = skipped + 1

    print('Skipped Particles : ', skipped)

    savename_detections_image = filename[:-4] + '_Detections.png'
    fig.savefig(savename_detections_image, dpi=figdpi, orientation='portrait', transparent=False, frameon=False)

    # count number of particles and fibers
    num_all_objects, num_fibers, num_particles, num_ND = pa.count_objecttypes(objects)

    # calculate number of needed columns and rows for particle grid plot
    nr, nc = pa.calc_grid(num_all_objects)

    # create list with all particle IDs
    ids = list(objects)
    print('All Objects :', num_all_objects)
    print('Particles   :', num_particles)
    print('Fibers      :', num_fibers)
    print('Not Defined :', num_ND)

    # get all IDs for particles detected as fibers
    fiber_ids = pa.find_fibers(objects)
    print('Fiber IDs: ', fiber_ids)

    if len(fiber_ids) > 0:
        id2show = fiber_ids[0]
    if len(fiber_ids) == 0:
        id2show = ids[0]

    obj = objects[id2show]

    print('Class                : ', obj['Class'])
    print('Area         [micron]:  {:.2f}'.format(obj['Area']))
    print('FeretMax     [micron]:  {:.2f}'.format(obj['FeretMax']))
    print('FeretMin     [micron]:  {:.2f}'.format(obj['FeretMin']))
    print('FeretRatio           :  {:.2f}'.format(obj['FeretRatio']))
    print('FiberLength  [micron]:  {:.2f}'.format(obj['FiberLength']))
    print('IsFiber              : ', obj['IsFiber'])
    print('Perimeter    [micron]:  {:.2f}'.format(obj['Perimeter']))
    print('Diameter MIC [micron]:  {:.2f}'.format(obj['maxMIC']))

    fig, axobj = plt.subplots(figsize=(8, 8))

    if obj['IsFiber']:
        axobj.imshow(obj['IntensityImage'], cmap=plt.cm.hot)
    if not obj['IsFiber']:
        axobj.imshow(obj['IntensityImage'], cmap='viridis')

    axobj.set_facecolor('grey')
    axobj.set_title('Particle ID:' + str(id2show) + ' - Class:' + obj['Class'] + ' - IsFiber:' + str(obj['IsFiber']))

    savename_pa_fiber_image = filename[:-4] + '_ID=' + str(id2show) + '.png'
    fig.savefig(savename_pa_fiber_image, dpi=100, orientation='portrait', transparent=False, frameon=False)
    plt.close(fig)

    maxcols = 15

    # create subplots
    fig, axp = plt.subplots(nrows=nr, ncols=maxcols, figsize=(16, 16), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(left=0.03, right=0.97, hspace=0.5, wspace=0.05)

    count = 0

    for axpp, pid in zip(axp.flat, ids):

        count = count + 1

        # extract the image of the current particle
        p = objects[pid]['IntensityImage']
        c = objects[pid]['Class']

        if objects[pid]['IsFiber']:
            axpp.imshow(p, cmap=plt.cm.hot, interpolation='nearest')

        if not objects[pid]['IsFiber']:
            axpp.imshow(p, cmap='viridis', interpolation='nearest')

        axpp.set_title(str(pid) + ' - ' + c)
        axpp.set_facecolor('white')

    # remove empty axes
    for ad in range(num_particles, nr * maxcols):
        fig.delaxes(axp.flatten()[ad])

    savename_grid_image = filename[:-4] + '_Grid.png'
    fig.savefig(savename_grid_image, dpi=figdpi, orientation='portrait', transparent=False, frameon=False)
    plt.close(fig)

    # create list for a specific parameter
    parameters = []

    for k in objects.keys():
        parameters.append(objects[k][pa_parameter])

    param_array = np.asarray(parameters)

    # create figure
    fig, ax_area = plt.subplots(1, 1, figsize=(12, 8))

    # Display histogram and return values
    p_values, p_bins, p_bars = ax_area.hist(param_array.ravel(),
                                            bins='auto',
                                            # bins=256,
                                            align='mid',
                                            rwidth=0.5,
                                            log=False,
                                            color='red',
                                            label=pa_parameter)

    # find the index for the highest values
    v_param = np.where(p_values == p_values.max())

    # get the most frequent pixel value
    p_most_frequent_value = pa.findhistogrammpeak(p_values, p_bins)

    p_mean = np.round(param_array.mean(), 1)
    p_median = np.round(np.median(param_array), 1)
    p_std = np.round(param_array.std(), 1)

    #ax_area.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_area.set_xlabel(pa_parameter, fontsize=16)
    ax_area.set_ylabel('Frequency', fontsize=16)
    ax_area.set_xlim(0, param_array.max())
    ax_area.grid(True)
    ax_area.legend(loc='center right')

    # plot vertical lines for mean an median values
    ax_area1 = ax_area.twinx()
    ax_area1.vlines(p_mean, 0, 1, colors='b', linestyles='dashed', lw=5, label='Mean')
    ax_area1.vlines(p_median, 0, 1, colors='g', linestyles='dashed', lw=5, label='Median')
    ax_area1.set_ylim(0, 1)
    ax_area1.legend(loc='upper right')
    # Turn off tick labels
    ax_area1.set_yticklabels([])

    savename_distribution_image = filename[:-4] + '_' + pa_parameter + '_Dist.png'
    fig.savefig(savename_distribution_image, dpi=figdpi, orientation='portrait', transparent=False, frameon=False)
    plt.close(fig)

    print('Most frequent Value       : ', p_most_frequent_value)
    print('Index for this Value      : ', v_param[0])
    print('Minimum Particle Size     :  {:.2f}'.format(param_array.min()))
    print('Maximum Particle Size     :  {:.2f}'.format(param_array.max()))
    print('Mean Particle Size        :  {:.2f}'.format(p_mean))
    print('Median Particle Size      :  {:.2f}'.format(p_median))
    print('Std. Dev. Particle Size   :  {:.2f}'.format(p_std))

    # create pandas dataframe with all particles
    df = pd.DataFrame(objects)
    df = df.transpose().reset_index()
    df = df.drop(['ClassColor', 'LabelImage', 'IntensityImage'], axis=1)
    df.rename(index=str, columns={"index": "ParticleID"}, inplace=True)

    # define new order of columns
    columnsTitles = ['ParticleID',
                     'Class',
                     'Area',
                     'IsFiber',
                     'FiberLength',
                     'FeretMin',
                     'FeretMax',
                     'FeretRatio',
                     'maxMIC',
                     'Perimeter']

    # do the re-indexing and show the dataframe
    df = df.reindex(columns=columnsTitles)
    print(df[:5])

    # display the result for the particle classes and their respective counts
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax = sns.countplot(x="Class", data=df)
    ax.grid(True)

    ax.set_title('Particle Class Distribution: ' + os.path.basename(filename), fontsize=18)
    ax.set_xlabel('Class', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

    savename_classcount_image = filename[:-4] + '_ClassCounts.png'
    fig.savefig(savename_classcount_image, dpi=figdpi, orientation='portrait', transparent=False, frameon=False)
    plt.close(fig)

    # get filename without extension
    basename_woext = os.path.basename(filename).split('.')[0]
    basepath = os.path.dirname(filename)

    # define name for excelsheet and CSV table
    pa_results_xlsx = os.path.join(basepath, basename_woext + '_PA_Results.xlsx')
    pa_results_csv = os.path.join(basepath, basename_woext + '_PA_Results.csv')

    writer = pd.ExcelWriter(pa_results_xlsx)
    df.to_excel(writer, 'Particle Analysis Results')
    writer.save()
    print('Saved results Excel :', pa_results_xlsx)

    df.to_csv(pa_results_csv, index=False, header=True, decimal='.', sep=separator)
    print('Saved results CSV :', pa_results_csv)

    outputs = {}
    outputs['histogram_image'] = savename_histogram_image
    outputs['threshold_image'] = savename_threshold_image
    outputs['detections_image'] = savename_detections_image
    outputs['pa_fiber_image'] = savename_pa_fiber_image
    outputs['grid_image'] = savename_grid_image
    outputs['distribution_image'] = savename_distribution_image
    outputs['classcount_image'] = savename_classcount_image
    outputs['pa_results_xlsx'] = pa_results_xlsx
    outputs['pa_results_csv'] = pa_results_csv

    print('Done.')

    return outputs
