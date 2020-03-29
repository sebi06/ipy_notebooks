from aicsimageio import AICSImage, imread, imread_dask
import numpy as np
#import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.segmentation import random_walker
from skimage.filters import threshold_otsu, threshold_triangle, rank
from skimage.segmentation import clear_border
from skimage.color import label2rgb
from skimage.util import invert
from skimage.filters import median, gaussian
from skimage.morphology import closing, square
from skimage.morphology import remove_small_objects, remove_small_holes, disk
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
        thresh = value
        
    if method == 'triangle':
        thresh = threshold_triangle(image2d)
        
    binary = image2d >= thresh

    return binary


def count_objects(image2d):
    
    image2d = image2d[600:900, 400:700]
    
    # filter image
    #image2d = median(image2d, selem=disk(3))
    image2d = gaussian(image2d, sigma=2, mode='reflect')
    
    binary = autoThresholding(image2d, method='triangle')
    distance = ndimage.distance_transform_edt(binary)
    
    #distance[distance < 1] = 0
    
    local_maxi = peak_local_max(distance,
                                #min_distance=3,
                                indices=False,
                                footprint=np.ones((13,13)),
                                labels=binary)
    
    markers, num_features = ndimage.label(local_maxi)
    
    labels = watershed(-distance, markers, mask=binary, watershed_line=True)
    image_label_overlay = label2rgb(labels, image=image2d, bg_label=0)
    
    regions = regionprops(labels)
    regions = [r for r in regions if r.area > 50]

    number_of_objects = len(regions) - 1
    
    # display the result
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    ax[0, 0].imshow(image2d, cmap=plt.cm.gray, interpolation='nearest')
    ax[0, 1].imshow(binary)
    ax[1, 0].imshow(distance)
    ax[1, 1].imshow(image_label_overlay)
    #ax[1, 1].imshow(labels)

    ax[0, 0].set_title('Original', fontsize=12)
    ax[0, 1].set_title('Binary', fontsize=12)
    ax[1, 0].set_title('Distance Map', fontsize=12)
    ax[1, 1].set_title('Labels', fontsize=12)
    """
    # display the result
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image2d, cmap=plt.cm.gray, interpolation='nearest')
    """
 
    
    for region in regionprops(labels):
       # take regions with large enough areas
       if region.area >= 100:
           # draw rectangle around segmented coins
           minr, minc, maxr, maxc = region.bbox
           rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                     fill=False, edgecolor='red', linewidth=2)
           ax[0,0].add_patch(rect)
           #ax.add_patch(rect)
    
    #plt.show()

    return number_of_objects, labels


#filename = r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\testwell96.czi'
filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate/testwell96.czi'


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


