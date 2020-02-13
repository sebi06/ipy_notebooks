import numpy as np
from aicspylibczi import CziFile
from pathlib import Path
import matplotlib.pyplot as plt

pth = Path('20190610_S02-02.czi')
czi = CziFile(pth)

# Get the shape of the data, the coordinate pairs are (start index, size)
dimensions = czi.dims_shape()  # [{'X': (0, 1900), 'Y': (0, 1300), 'Z': (0, 60), 'C': (0, 4), 'S': (0, 40), 'B': (0, 1)}]

czi.dims  # BSCZYX

czi.size  # (1, 40, 4, 60, 1300, 1900)

# Load the image slice I want from the file
img, shp = czi.read_image(S=13, Z=16)

# shp = [('B', 1), ('S', 1), ('C', 4), ('Z', 1), ('Y', 1300), ('X', 1900)]  # List[(Dimension, size), ...]
# img.shape = (1, 1, 4, 1, 1300, 1900)   # numpy.ndarray

# define helper functions


def norm_by(x, min_, max_):
    norms = np.percentile(x, [min_, max_])
    i2 = np.clip((x - norms[0]) / (norms[1] - norms[0]), 0, 1)
    return i2


def recolor(im):  # transform from rgb to cyan-magenta-yellow
    im_shape = np.array(im.shape)
    color_transform = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]).T
    im_reshape = im.reshape([np.prod(im_shape[0:2]), im_shape[2]]).T
    im_recolored = np.matmul(color_transform.T, im_reshape).T
    im_shape[2] = 3
    im = im_recolored.reshape(im_shape)
    return im


# normalize, combine into RGB and transform to CMY
c1 = (norm_by(img[0, 0, 0, 0, 0:750, 250:1000], 50, 99.8) * 255).astype(np.uint8)
c2 = (norm_by(img[0, 0, 1, 0, 0:750, 250:1000], 50, 99.8) * 255).astype(np.uint8)
c3 = (norm_by(img[0, 0, 2, 0, 0:750, 250:1000], 0, 100) * 255).astype(np.uint8)
rgb = np.stack((c1, c2, c3), axis=2)
cmy = np.clip(recolor(rgb), 0, 255)

# plot using matplotlib¶
plt.figure(figsize=(10, 10))
plt.imshow(cmy)
plt.axis('off')
