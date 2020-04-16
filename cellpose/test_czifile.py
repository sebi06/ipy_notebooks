import czifile as zis
import numpy as np
from matplotlib import pyplot as plt, cm
from aicsimageio import AICSImage


#filename = r'c:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Castor\testwell96.czi'
filename = r"C:\Users\m1srh\Downloads\Experiment-06.czi"

# Get an AICSImage object
#img = AICSImage(filename)
# md = img.metadata  # returns the metadata object for this image type

# get CZI object and read array
czi = zis.CziFile(filename)
mdczi = czi.metadata(raw=False)

subblockdir = czi.filtered_subblock_directory

count = 0

for sb in subblockdir:

    count += 1
    datasegment = sb.data_segment()
    data = datasegment.data()

    Mindex = datasegment.dimension_entries[0].size
    Bindex = datasegment.dimension_entries[1].size
    Sindex = datasegment.dimension_entries[2].size
    Cindex = datasegment.dimension_entries[3].size

    m = datasegment.directory_entry.mosaic_index

    #fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    print(count, Cindex)
    # print(m)

    """
    img = data[0, 0, 0, :, :, 0]

    
    ax.imshow(img,
              cmap=plt.cm.gray,
              interpolation='nearest',
              clim=[img.min(), img.max() * 0.5])

    plt.show()
    """


czi.close()
