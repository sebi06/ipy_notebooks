from aicsimageio import AICSImage, imread
import imgfileutils as imf

# get list of all filenames
#filename = r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\testwell96.czi'

filename = r"C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\WP384_2CH_4Pos_A4-10_DAPI_GFP.czi"
#filename = r"C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\WP384_2CH_4Pos_A4-10_DAPI_GFP_C1.czi"
#filename = r"C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\WP384_2CH_4Pos_A4-10_DAPI_GFP_C2.czi"


# parse the CZI metadata return the metadata dictionary and additional information
metadata = imf.get_metadata_czi(filename, dim2none=False)

print('MD from czifile.py - Shape: ', metadata['Shape'])
print('MD from czifile.py - Axes: ', metadata['Axes'])
print('MD from aics - Shape: ', metadata['Shape_aics'])
print('MD from aics - Axes: ', metadata['Axes_aics'])


# Get an AICSImage object
#img = AICSImage(filename)
img = AICSImage(filename)

#for k, v in metadata.items():
#    print(k, v)

# read specific scene
#scenes = img.get_image_dask_data("CYX", S=0, T=0, Z=0)
#scene_array = scene.compute()


#img.view_napari()

img.close()

print('Done')
