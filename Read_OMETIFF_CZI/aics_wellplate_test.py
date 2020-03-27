from aicsimageio import AICSImage, imread, imread_dask
import numpy as np
#import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.segmentation import clear_border
from skimage.color import label2rgb
from skimage.util import invert
from skimage.morphology import closing, square
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops
from skimage.segmentation import random_walker
from scipy import ndimage
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
#from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
#from skimage import data
#from skimage import img_as_float
#from skimage.morphology import reconstruction
#import skimage
#from skimage import segmentation


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
        binary = image2d >= value
        
    if method == 'triangle':
        thresh = threshold_triangle(image2d)
        
    binary = image2d >= thresh

    return binary


def count_objects(image2d):
    
    image2d = image2d[400:900, 300:800]
    
    
    binary = autoThresholding(image2d, method='triangle')
    distance = ndimage.distance_transform_edt(binary)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((5, 5)), labels=binary)
    markers, num_features = ndimage.label(local_maxi)
    labels = watershed(-distance, markers, mask=binary)
    
    image_label_overlay = label2rgb(labels, image=image2d, bg_label=0)
    
    regions = regionprops(labels)
    regions = [r for r in regions if r.area > 50]

    number_of_objects = len(regions) - 1
    
    # display the result
    fig, ax = plt.subplots(2, 2, figsize=(16, 8))

    ax[0, 0].imshow(image2d, cmap=plt.cm.gray, interpolation='nearest')
    ax[0, 1].imshow(distance)
    ax[1, 0].imshow(local_maxi)
    ax[1, 1].imshow(image_label_overlay)

    ax[0, 0].set_title('Original', fontsize=12)
    ax[0, 1].set_title('Distance Map', fontsize=12)
    ax[1, 0].set_title('LocalMaxOriginal', fontsize=12)
    ax[1, 1].set_title('Labels', fontsize=12)
 
    
    for region in regionprops(labels):
       # take regions with large enough areas
       if region.area >= 100:
           # draw rectangle around segmented coins
           minr, minc, maxr, maxc = region.bbox
           rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                     fill=False, edgecolor='red', linewidth=2)
           ax[0,0].add_patch(rect)
    
    #plt.show()

    return number_of_objects, labels


filename = r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\testwell96.czi'


# Get an AICSImage object
img = AICSImage(filename)

cells = []

for s in range(1):
#for s in range(img.size_s):
    
    print('Analyzing Scene : ', s)
    data = img.get_image_data("YX", S=s, T=0, Z=0, C=0)
    num_obj, labelimage = count_objects(data) 
    cells.append(num_obj)

img.close()

print(cells)
print('Done.')


