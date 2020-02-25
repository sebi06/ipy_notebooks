from aicsimageio import AICSImage, imread
import imgfileutils as imf

# get list of all filenames
#filename = r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\testwell96.czi'
filename = r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\384well_DAPI.czi'

# Get an AICSImage object
#img = AICSImage(filename)
img = AICSImage(filename, chunk_by_dims=["S"])


# show the dimensions

print(img.dims)
print(img.shape)

metadata = {}

metadata['SizeX'] = img.size_x
metadata['SizeY'] = img.size_y
metadata['SizeC'] = img.size_c
metadata['SizeZ'] = img.size_t
metadata['SizeT'] = img.size_t
metadata['SizeS'] = img.size_s

for k, v in metadata.items():
    print(k, v)

# read specific scene
#scenes = img.get_image_dask_data("CYX", S=0, T=0, Z=0)
#scene_array = scene.compute()


img.view_napari()

print('Done')
