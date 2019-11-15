
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import czifile as zis
from apeer_ometiff_library import io, processing  # , omexmlClass
import os
from matplotlib import pyplot as plt, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.viewer import ImageViewer
import skimage.io
import matplotlib.colors as colors
import numpy as np
#from skimage.external import tifffile
import ipywidgets as widgets

import imgfileutils as imf
import zarr


# define your testfiles here
testfolder = r'C:\Users\m1srh\Documents\GitHub\ipy_notebooks\Read_OMETIFF_CZI\testdata'

imgdict = {
    1: os.path.join(testfolder, r'CellDivision_T=10_Z=15_CH=2_DCV_small_green.ome.tiff'),
    2: os.path.join(testfolder, r'CellDivision_T=10_Z=15_CH=2_DCV_small_red.ome.tiff'),
    3: os.path.join(testfolder, r'CellDivision_T=10_Z=15_CH=2_DCV_small.ome.tiff'),
    4: os.path.join(testfolder, r'CellDivision_T=10_Z=15_CH=2_DCV_small_green.czi'),
    5: os.path.join(testfolder, r'CellDivision_T=10_Z=15_CH=2_DCV_small_red.czi'),
    6: os.path.join(testfolder, r'CellDivision_T=10_Z=15_CH=2_DCV_small.czi')
}

filename = imgdict[6]
image_name = os.path.basename(filename)

if filename.lower().endswith('.ome.tiff') or filename.lower().endswith('.ome.tif'):
    # Return value is an array of order (T, Z, C, X, Y)
    (array, omexml) = io.read_ometiff(filename)
    metadata = imf.get_metadata(filename, series=0)

if filename.lower().endswith('.czi'):
    array, metadata = imf.get_array_czi(filename, replacezero=False)
    print(metadata['Shape'])
    print(metadata['Axes'])
    print(array.shape)

z = zarr.array(array, chunks=(1, 1, 2, 15, 256, 256), dtype='int16')
print(z.info)


if filename.lower().endswith('.ome.tiff') or filename.lower().endswith('.ome.tif'):
    ui, out = imf.create_ipyviewer_ome_tiff(array, metadata)

if filename.lower().endswith('.czi'):
    ui, out = imf.create_ipyviewer_czi(array, metadata)


# try to configre napari automatiaclly based on metadata
imf.show_napari(array, metadata)


# for k, v in metadata.items():
#
#    print(k, v)
