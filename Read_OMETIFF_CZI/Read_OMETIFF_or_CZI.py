
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
import ipywidgets as widgets

import imgfileutils as imf


imgdict1 = {
    1: r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/testczi/XYCZ-Regions-T_CH=2_Z=5_T=3_Tile=2x2.czi',
    2: r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/testczi/XYCZ-Regions-T_CH=2_Z=5_Tile=2x2_T=3.czi',
    3: r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/testczi/XYCZT_CH=2_Z=5_All_CH_per_Slice.czi',
    4: r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/testczi/XYZCT_Z=5_CH=2_Z=5_FullStack_per_CH.czi',
    5: r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/testczi/XYZCT_Z=15_C=2_T=20',
    6: r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/testczi/2x2_SNAP_CH=2_Z=5_T=2.czi',
    7: r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/testczi/S=2_T=5_Z=3_CH=2_A2.czi',
    8: r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/celldivison/CellDivision_T=10_Z=20_CH=1_DCV.czi',
    9: r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/celldivisonCellDivision_T=15_Z=20_CH=2_DCV.czi',
    10: r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/particles/Filter_with_Particles_small.czi',
    11: r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/Brainslide/BrainProject/8Brains_DAPI_5X_stitched.czi',
    12: r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/Brainslide/BrainProject/DTScan_ID3.czi',
    13: r'/datadisk1/tuxedo/testpictures/Fruit_Fly_Brain_3D/Fruit_Fly_Brain.ome.tif',
    14: r'/datadisk1/tuxedo/testpictures/Fruit_Fly_Brain_3D/Fruit_Fly_Brain.ome.czi',
    15: r'c:\Users\m1srh\Documents\Testdata_Zeiss\AxioScan\kungel_RGB.czi',
    16: r'c:\Users\m1srh\Documents\Testdata_Zeiss\AxioScan\kungel_RGB_comp2.czi',
    17: r'C:\Temp\input\Filter_with_Particles_small.ome.tiff',
    18: r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\Z-Stack_DCV\CellDivision_T=10_Z=15_CH=2_DCV_small.czi',
    19: r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\Z-Stack_DCV\CellDivision_T=10_Z=15_CH=2_DCV_small.ome.tiff',
    20: r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\Z-Stack_DCV\CellDivision_T=10_Z=15_CH=2_DCV_small_Fiji.ome.tiff',
    21: r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\Z-Stack_DCV\CellDivision_T=15_Z=20_CH=2_DCV.czi',
    22: r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\Z-Stack_DCV\NeuroSpheres_DCV_A635_A488_A405.czi',
    23: r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\Z-Stack_DCV\NeuroSpheres_DCV_A635_A488_A405.ome.tiff',
    24: r'E:\tuxedo\testpictures\Testdata_Zeiss\em\FIB_Stack.czi',
    25: r'C:\Users\m1srh\Documents\Testdata_Zeiss\Cryo\FIB_Stack.czi',
    26: r'E:\tuxedo\testpictures\Testdata_Zeiss\celldivison\CellDivision_T=15_Z=20_CH=2_DCV.czi',
    27: r'C:\Users\m1srh\Documents\Testdata_Zeiss\Fruit_Fly_Brain\Fruit_Fly_Brain_3D.czi',
    28: r'C:\Users\m1srh\Documents\Testdata_Zeiss\Fruit_Fly_Brain\Fruit_Fly_Brain_3D.ome.tiff'
}

# define your testfiles here

testfolder = r'C:\Users\m1srh\Documents\GitHub\ipy_notebooks\Read_OMETIFF_CZI\testdata'

imgdict2 = {
    1: os.path.join(testfolder, r'CellDivision_T=10_Z=15_CH=2_DCV_small_green.ome.tiff'),
    2: os.path.join(testfolder, r'CellDivision_T=10_Z=15_CH=2_DCV_small_red.ome.tiff'),
    3: os.path.join(testfolder, r'CellDivision_T=10_Z=15_CH=2_DCV_small.ome.tiff'),
    4: os.path.join(testfolder, r'CellDivision_T=10_Z=15_CH=2_DCV_small_green.czi'),
    5: os.path.join(testfolder, r'CellDivision_T=10_Z=15_CH=2_DCV_small_red.czi'),
    6: os.path.join(testfolder, r'CellDivision_T=10_Z=15_CH=2_DCV_small.czi')
}

#filename = imgdict2[6]

filename = imgdict1[28]

image_name = os.path.basename(filename)

if filename.lower().endswith('.ome.tiff') or filename.lower().endswith('.ome.tif'):

    # Return value is an array of order (T, Z, C, X, Y)
    (array, omexml) = io.read_ometiff(filename)
    metadata = imf.get_metadata(filename, series=0)
    #ui, out = imf.create_ipyviewer_ome_tiff(array, metadata)

if filename.lower().endswith('.czi'):

    array, metadata = imf.get_array_czi(filename, replacezero=False)
    print(metadata['Shape'])
    print(metadata['Axes'])
    print(array.shape)

    #ui, out = imf.create_ipyviewer_czi(array, metadata)

print('Array Shape: ', array.shape)

# try to configure napari automatically based on metadata
imf.show_napari(array, metadata)


# for k, v in metadata.items():
#
#    print(k, v)
