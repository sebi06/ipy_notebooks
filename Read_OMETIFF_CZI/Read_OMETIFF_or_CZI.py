
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import czifile as zis
from apeer_ometiff_library import io, processing, omexmlClass
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


#basefolder = r'/datadisk1/tuxedo/IPython_Notebooks/testdata'
#basefolder = r'/home/sebi06/testdata'
basefolder = r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\Z-Stack_DCV'

#filename = os.path.join(basefolder, 'Filter_with_Particles_big.ome.tiff')
#filename = os.path.join(basefolder, 'S=2_T=5_CH=3_CH=2_A2.ome.tiff')
#filename = os.path.join(basefolder, 'Osteosarcoma_01.ome.tiff')
#filename = os.path.join(basefolder, 'Filter_with_Particles_small.czi')
#filename = os.path.join(basefolder, '8Brains_DAPI_5X_stitched.czi')
#filename = os.path.join(basefolder, r'2x2_SNAP_CH=2_Z=5_T=2.czi')
#filename = os.path.join(basefolder, 'S=2_T=5_Z=3_CH=2_A2.czi')
filename = os.path.join(basefolder, r'CellDivision_T=15_Z=20_CH=2_DCV.czi')


image_name = os.path.basename(filename)

image_name = os.path.basename(filename)

if filename.lower().endswith('.ome.tiff') or filename.lower().endswith('.ome.tif'):

    # Return value is an array of order (T, Z, C, X, Y)
    (array, omexml) = io.read_ometiff(filename)
    metadata = imf.get_metadata(filename, series=0)

if filename.lower().endswith('.czi'):

    metadata = imf.get_metadata(filename)

    print(metadata['Shape'])
    print(metadata['Axes'])

    array, dim_dict = imf.get_array_czi(filename,
                                        cziaxes=metadata['Axes'],
                                        blockindex=0,
                                        sceneindex=0,
                                        replacezero=False)

    metadata['DimOrder CZI'] = dim_dict
    print(array.shape)

print(metadata['Shape'])
print(metadata['Axes'])
print(array.shape)
print(metadata['Extension'])

if metadata['ImageType'] == 'ometiff':
    ui, out = imf.create_ipyviewer_ometiff(array, metadata)
if metadata['ImageType'] == 'czi':
    ui, out = imf.create_ipyviewer_czi(array, metadata)

for k, v in metadata.items():

    print(k, v)
