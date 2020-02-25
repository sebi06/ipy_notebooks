from aicsimageio import AICSImage, imread_dask

# Get an AICSImage object
img = AICSImage("my_file.tiff")
img.dask_data  # returns 6D STCZYX dask array
img.dims  # returns string "STCZYX"
img.shape  # returns tuple of dimension sizes in STCZYX order
img.size("STC")  # returns tuple of dimensions sizes for just STC
img.get_image_dask_data("CZYX", S=0, T=0)  # returns 4D CZYX dask array

# Read specified portion of dask array
lazy_s0t0 = img.get_image_dask_data("CZYX", S=0, T=0)  # returns 4D CZYX dask array
s0t0 = lazy_s0t0.compute()  # returns 4D CZYX numpy array

# Or use normal numpy array slicing
lazy_data = imread_dask("my_file.tiff")
lazy_s0t0 = lazy_data[0, 0, :]
s0t0 = lazy_s0t0.compute()
