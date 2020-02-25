from aicsimageio import AICSImage

# Get an AICSImage object
img = AICSImage("my_file.tiff")
img.view_napari()  # launches napari GUI and viewer
