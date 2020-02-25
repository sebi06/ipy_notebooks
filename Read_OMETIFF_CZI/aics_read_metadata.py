from aicsimageio import AICSImage

# Get an AICSImage object
img = AICSImage("my_file.tiff")
img.metadata  # returns the metadata object for this image type
img.get_channel_names()  # returns a list of string channel names if found in the metadata
