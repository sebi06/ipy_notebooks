from aicsimageio import AICSImage, imread


filename = r'testdata/CellDivision_T=10_Z=15_CH=2_DCV_small.ome.tiff'

# Get an AICSImage object
img = AICSImage(filename)
array6d = img.data  # returns 6D STCZYX numpy array
dims = img.dims  # returns string "STCZYX"
array_shape = img.shape  # returns tuple of dimension sizes in STCZYX order
dims_subarray = img.size("STC")  # returns tuple of dimensions sizes for just STC
subarray = img.get_image_data("CZYX", S=0, T=0)  # returns 4D CZYX numpy array

# get metadata
md = img.metadata  # returns the metadata object for this image type


# Get 6D STCZYX numpy array
#data = imread(filename)

print('Done.')

img.view_napari()  # launches napari GUI and viewer
