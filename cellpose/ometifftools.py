import javabridge as jv
import bioformats
import tifffile
import os
import numpy as np
from aicsimageio import AICSImage, imread
from aicsimageio.writers import ome_tiff_writer


def write_ometiff(filepath, img,
                  scalex=0.1,
                  scaley=0.1,
                  scalez=1.0,
                  dimorder='TZCYX',
                  pixeltype=np.uint16,
                  swapxyaxes=True,
                  series=1):
    """
    This function will write an OME-TIFF file to disk.
    The out 6D array has the following dimension order:

    [T, Z, C, Y, X] if swapxyaxes = True

    [T, Z, C, X, Y] if swapxyaxes = False
    """

    # Dimension STZCXY
    if swapxyaxes:
        # swap xy to write the OME-Stack with the correct shape
        SizeT = img.shape[0]
        SizeZ = img.shape[1]
        SizeC = img.shape[2]
        SizeX = img.shape[4]
        SizeY = img.shape[3]

    if not swapxyaxes:
        SizeT = img.shape[0]
        SizeZ = img.shape[1]
        SizeC = img.shape[2]
        SizeX = img.shape[3]
        SizeY = img.shape[4]

    # Getting metadata info
    omexml = bioformats.omexml.OMEXML()
    omexml.image(series - 1).Name = filepath

    for s in range(series):
        p = omexml.image(s).Pixels
        p.ID = str(s)
        p.SizeX = SizeX
        p.SizeY = SizeY
        p.SizeC = SizeC
        p.SizeT = SizeT
        p.SizeZ = SizeZ
        p.PhysicalSizeX = np.float(scalex)
        p.PhysicalSizeY = np.float(scaley)
        p.PhysicalSizeZ = np.float(scalez)
        if pixeltype == np.uint8:
            p.PixelType = 'uint8'
        if pixeltype == np.uint16:
            p.PixelType = 'uint16'
        p.channel_count = SizeC
        p.plane_count = SizeZ * SizeT * SizeC
        p = writeOMETIFFplanes(p, SizeT=SizeT, SizeZ=SizeZ, SizeC=SizeC, order=dimorder)

        for c in range(SizeC):
            # if pixeltype == 'unit8':
            if pixeltype == np.uint8:
                p.Channel(c).SamplesPerPixel = 1

            if pixeltype == np.uint16:
                p.Channel(c).SamplesPerPixel = 2

        omexml.structured_annotations.add_original_metadata(bioformats.omexml.OM_SAMPLES_PER_PIXEL, str(SizeC))

    # Converting to omexml
    xml = omexml.to_xml(encoding='utf-8')

    # write file and save OME-XML as description
    tifffile.imwrite(filepath, img, metadata={'axes': dimorder}, description=xml)

    return filepath


def writeOMETIFFplanes(pixel, SizeT=1, SizeZ=1, SizeC=1, order='TZCXY', verbose=False):

    if order == 'TZCYX' or order == 'TZCXY':

        pixel.DimensionOrder = bioformats.omexml.DO_XYCZT
        counter = 0
        for t in range(SizeT):
            for z in range(SizeZ):
                for c in range(SizeC):

                    if verbose:
                        print('Write PlaneTable: ', t, z, c),
                        sys.stdout.flush()

                    pixel.Plane(counter).TheT = t
                    pixel.Plane(counter).TheZ = z
                    pixel.Plane(counter).TheC = c
                    counter = counter + 1

    return pixel


def write_ometiff_aicsimageio(filepath, imgarray, metadata, overwrite=False):

    # define scaling
    try:
        pixels_physical_size = [metadata['XScale'],
                                metadata['YScale'],
                                metadata['ZScale']]
    except KeyError as e:
        print('Key not found:', e)
        pixels_physical_size = None

    # define channel names list
    try:
        channel_names = []
        for ch in metadata['Channels']:
            channel_names.append(ch)
    except KeyError as e:
        print('Key not found:', e)
        channel_names = None

    # define correct string fro dimension order
    if len(imgarray.shape) == 5:
        dimension_order = 'TZCYX'
    elif len(imgarray.shape) == 4:
        dimension_order = 'ZCYX'
    elif len(imgarray.shape) == 3:
        dimension_order = 'CYX'
    elif len(imgarray.shape) == 2:
        dimension_order = 'YX'

    # write the array as an OME-TIFF incl. the metadata
    with ome_tiff_writer.OmeTiffWriter(filepath, overwrite_file=overwrite) as writer:
        writer.save(imgarray,
                    channel_names=channel_names,
                    image_name=os.path.basename((filepath)),
                    pixels_physical_size=pixels_physical_size,
                    channel_colors=None,
                    dimension_order=dimension_order)

    print('Saved image as: ', filepath)
