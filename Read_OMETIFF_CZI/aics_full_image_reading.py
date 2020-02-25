from aicsimageio import AICSImage, imread

# Get an AICSImage object
img = AICSImage("my_file.tiff")
img.data  # returns 6D STCZYX numpy array
img.dims  # returns string "STCZYX"
img.shape  # returns tuple of dimension sizes in STCZYX order
img.size("STC")  # returns tuple of dimensions sizes for just STC
img.get_image_data("CZYX", S=0, T=0)  # returns 4D CZYX numpy array

# Get 6D STCZYX numpy array
data = imread("my_file.tiff")
