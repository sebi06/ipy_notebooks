import numpy as np
import aicspylibczi
import pathlib
from PIL import Image


mosaic_file = pathlib.Path('mosaic_test.czi')
czi = aicspylibczi.CziFile(mosaic_file)

# Get the shape of the data
dimensions = czi.dims  # 'STCZMYX'

czi.size  # (1, 1, 1, 1, 2, 624, 924)

czi.dims_shape()  # [{'X': (0, 924), 'Y': (0, 624), 'Z': (0, 1), 'C': (0, 1), 'T': (0, 1), 'M': (0, 2), 'S': (0, 1)}]

czi.is_mosaic()  # True
# Mosaic files ignore the S dimension and use an internal mIndex to reconstruct, the scale factor allows one to generate a manageable image
mosaic_data = czi.read_mosaic(C=0, scale_factor=1)

mosaic_data.shape  # (1, 1, 624, 1756)
# the C channel has been specified S & M are used internally for position so this is (T, Z, Y, X)

normed_mosaic_data = norm_by(mosaic_data[0, 0, :, :], 5, 98) * 255
img = Image.fromarray(normed_mosaic_data.astype(np.uint8))
