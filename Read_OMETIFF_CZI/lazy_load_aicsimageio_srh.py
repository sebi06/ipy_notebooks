
import imgfileutils as imf
from aicsimageio import AICSImage, imread, imread_dask
import numpy as np
import napari
import dask.array as da


def show_napari(array, metadata,
                blending='additive',
                gamma=0.85,
                verbose=True):
    """
    Show the multidimensional array using the Napari viewer

    :param array: multidimensional NumPy.Array containing the pixeldata
    :param metadata: dictionary with CZI or OME-TIFF metadata
    :param blending: NapariViewer option for blending
    :param gamma: NapariViewer value for Gamma
    :param verbose: show additional output
    """

    with napari.gui_qt():

        # create scalefcator with all ones
        scalefactors = [1.0] * len(array.shape)

        # initialize the napari viewer
        print('Initializing Napari Viewer ...')
        viewer = napari.Viewer()

        # find position of dimensions
        posZ = metadata['Axes'].find('Z')
        posC = metadata['Axes'].find('C')
        posT = metadata['Axes'].find('T')

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
                #channel = array.take(ch, axis=posC)
                channel = array[:, :, ch, :, :, :]
                print('Shape Channel : ', ch, channel.shape)

                # actually show the image array
                print('Adding Channel: ', chname)
                print('Scaling Factors: ', scalefactors)

                # get min-max values for initial scaling
                #clim = [da.min(channel), da.round(da.max(channel) * 0.85)]
                # if verbose:
                #    print('Scaling: ', clim)
                viewer.add_image(channel,
                                 name=chname,
                                 scale=scalefactors,
                                 # contrast_limits=clim,
                                 blending=blending,
                                 gamma=gamma)

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
                clim = [array.min(), np.round(array.max() * 0.85)]
                if verbose:
                    print('Scaling: ', clim)
                viewer.add_image(array,
                                 name=chname,
                                 scale=scalefactors,
                                 contrast_limits=clim,
                                 blending=blending,
                                 gamma=gamma)


imagefile = r'testdata/CellDivision_T=10_Z=15_CH=2_DCV_small.czi'

# parse the CZI metadata return the metadata dictionary and additional information
metadata = imf.get_metadata_czi(imagefile, dim2none=False)
additional_metadata = imf.get_additional_metadata_czi(imagefile)

# stack = AICSImage.get_image_dask_data(imagefile)
stack = imread_dask(imagefile)


# from dask_image.imread import imread

# stack = imread("/path/to/experiment/*.tif")
# with napari.gui_qt():
#    napari.view_image(stack, contrast_limits=[0, 2000], is_pyramid=False)


show_napari(stack, metadata,
            blending='additive',
            gamma=0.85,
            verbose=True)
