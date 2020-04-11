import os
from aicsimageio import AICSImage

filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/celldivision/CellDivision2_SF_deco.czi'
#filename = r'resources/CellDivision_T=10_Z=15_CH=2_DCV_small.czi'
#filename = r'resources/s_1_t_5_c_1_z_1.czi'

# Get an AICSImage object
img = AICSImage(filename)
img.view_napari()  # launches napari GUI and viewer
