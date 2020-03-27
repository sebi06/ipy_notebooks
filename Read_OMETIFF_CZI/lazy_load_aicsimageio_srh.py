
import imgfileutils as imf
from aicsimageio import AICSImage, imread, imread_dask
#import numpy as np
#import napari
#import dask.array as da


def show_napari(array, metadata,
                blending='additive',
                gamma=0.85,
                verbose=True,
                use_pylibczi=True):
    """
    Show the multidimensional array using the Napari viewer

    :param array: multidimensional NumPy.Array containing the pixeldata
    :param metadata: dictionary with CZI or OME-TIFF metadata
    :param blending: NapariViewer option for blending
    :param gamma: NapariViewer value for Gamma
    :param verbose: show additional output
    """

    def calc_scaling(data, corr_min=1.0,
                     offset_min=0,
                     corr_max=0.85,
                     offset_max=0):

        # get min-max values for initial scaling
        minvalue = np.round((data.min() + offset_min) * corr_min)
        maxvalue = np.round((data.max() + offset_max) * corr_max)
        print('Scaling: ', minvalue, maxvalue)

        return [minvalue, maxvalue]

    with napari.gui_qt():

        # create scalefcator with all ones
        scalefactors = [1.0] * len(array.shape)

        # initialize the napari viewer
        print('Initializing Napari Viewer ...')
        viewer = napari.Viewer()

        if not use_pylibczi:
            # use find position of dimensions
            posZ = metadata['Axes'].find('Z')
            posC = metadata['Axes'].find('C')
            posT = metadata['Axes'].find('T')

        if use_pylibczi:
            posZ = metadata['Axes_aics'].find('Z')
            posC = metadata['Axes_aics'].find('C')
            posT = metadata['Axes_aics'].find('T')

        # get the scalefactors from the metadata
        scalef = imf.get_scalefactor(metadata)
        # modify the tuple for the scales for napari
        scalefactors[posZ] = scalef['zx']

        if verbose:
            print('Dim PosT : ', posT)
            print('Dim PosZ : ', posZ)
            print('Dim PosC : ', posC)
            print('Scale Factors : ', scalefactors)

        if metadata['SizeC'] > 1:
            # add all channels as layers
            for ch in range(metadata['SizeC']):

                try:
                    # get the channel name
                    chname = metadata['Channels'][ch]
                except:
                    # or use CH1 etc. as string for the name
                    chname = 'CH' + str(ch + 1)

                # cut out channel
                # use dask if array is a dask.array
                if isinstance(array, da.Array):
                    channel = array.compute().take(ch, axis=posC)
                else:
                    # use normal numpy if not
                    channel = array.take(ch, axis=posC)

                print('Shape Channel : ', ch, channel.shape)

                # actually show the image array
                print('Adding Channel: ', chname)
                print('Scaling Factors: ', scalefactors)

                # get min-max values for initial scaling
                clim = calc_scaling(channel)

                viewer.add_image(channel,
                                 name=chname,
                                 scale=scalefactors,
                                 contrast_limits=clim,
                                 blending=blending,
                                 gamma=gamma,
                                 is_pyramid=False)

        if metadata['SizeC'] == 1:

            ch = 0
            # just add one channel as a layer
            try:
                # get the channel name
                chname = metadata['Channels'][ch]
            except:
                # or use CH1 etc. as string for the name
                chname = 'CH' + str(ch + 1)

            # actually show the image array
            print('Adding Channel: ', chname)
            print('Scaling Factors: ', scalefactors)

            # get min-max values for initial scaling
            clim = calc_scaling(array)

            viewer.add_image(array,
                             name=chname,
                             scale=scalefactors,
                             contrast_limits=clim,
                             blending=blending,
                             gamma=gamma,
                             is_pyramid=False)


#imagefile = r'testdata/CellDivision_T=10_Z=15_CH=2_DCV_small.czi'
imagefile = r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\testwell96.czi'
#imagefile = r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\384well_DAPI.czi'
#imagefile = r"C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\WP384_2CH_4Pos_A4-10_DAPI_GFP.czi"
#imagefile = r"C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\WP384_2CH_4Pos_A4-10_DAPI_GFP_C1.czi"
#imagefile = r"C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\WP384_2CH_4Pos_A4-10_DAPI_GFP_C2.czi"

# parse the CZI metadata return the metadata dictionary and additional information
metadata = imf.get_metadata_czi(imagefile, dim2none=False)
additional_metadata = imf.get_additional_metadata_czi(imagefile)

# option 1
#img = AICSImage(imagefile, chunk_by_dims=["S"])
#img = AICSImage(imagefile)
#stack = img.get_image_dask_data()

# option 2
stack = imread_dask(imagefile)

# option 3
stack = imread(imagefile)


# from dask_image.imread import imread

# stack = imread("/path/to/experiment/*.tif")
# with napari.gui_qt():
#    napari.view_image(stack, contrast_limits=[0, 2000], is_pyramid=False)


imf.show_napari(stack, metadata,
                blending='additive',
                gamma=0.85,
                verbose=True,
                use_pylibczi=True)
