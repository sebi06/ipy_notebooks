import czifile as zis
from apeer_ometiff_library import io, processing, omexmlClass
import os
#import cziutils as czt
from skimage.external import tifffile
import ipywidgets as widgets
from matplotlib import pyplot as plt, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xmltodict
import numpy as np


def create_metadata_dict():
    """
    A Python dictionary will be created to hold the relevant metadata.
    """

    metadata = {'Directory': None,
                'Filename': None,
                'Name': None,
                'AcqDate': None,
                'TotalSeries': None,
                'SizeX': None,
                'SizeY': None,
                'SizeZ': None,
                'SizeC': None,
                'SizeT': None,
                'DimOrder BF': None,
                'ObjNA': None,
                'ObjMag': None,
                'ObjID': None,
                'ObjName': None,
                'ObjImmersion': None,
                'XScale': None,
                'YScale': None,
                'ZScale': None,
                'XScaleUnit': None,
                'YScaleUnit': None,
                'ZScaleUnit': None,
                'DetectorModel': [],
                'DetectorName': [],
                'DetectorID': None,
                'InstrumentID': None,
                'Channels': [],
                'ImageIDs': []
                }

    return metadata


def get_metadata_ometiff(filename, omemd, series=0):

    # create dictionary for metadata and get OME-XML data
    metadata = create_metadata_dict()
    #md = omexmlClass.OMEXML(omexml)

    # get directory and filename etc.
    metadata['Directory'] = os.path.dirname(filename)
    metadata['Filename'] = os.path.basename(filename)
    metadata['AcqDate'] = omemd.image(series).AcquisitionDate
    metadata['Name'] = omemd.image(series).Name

    # get image dimensions
    metadata['SizeT'] = omemd.image(series).Pixels.SizeT
    metadata['SizeZ'] = omemd.image(series).Pixels.SizeZ
    metadata['SizeC'] = omemd.image(series).Pixels.SizeC
    metadata['SizeX'] = omemd.image(series).Pixels.SizeX
    metadata['SizeY'] = omemd.image(series).Pixels.SizeY
    # get number of series
    metadata['TotalSeries'] = omemd.get_image_count()

    # get dimension order
    metadata['DimOrder BF'] = omemd.image(series).Pixels.DimensionOrder

    # get the scaling
    metadata['XScale'] = omemd.image(series).Pixels.PhysicalSizeX
    metadata['XScaleUnit'] = omemd.image(series).Pixels.PhysicalSizeXUnit
    metadata['YScale'] = omemd.image(series).Pixels.PhysicalSizeY
    metadata['YScaleUnit'] = omemd.image(series).Pixels.PhysicalSizeYUnit
    metadata['ZScale'] = omemd.image(series).Pixels.PhysicalSizeZ
    metadata['ZScaleUnit'] = omemd.image(series).Pixels.PhysicalSizeZUnit

    # get all image IDs
    for i in range(omemd.get_image_count()):
        metadata['ImageIDs'].append(i)

    # get information about the instrument and objective
    metadata['InstrumentID'] = omemd.instrument(series).get_ID()
    metadata['DetectorModel'] = omemd.instrument(series).Detector.get_Model()
    metadata['DetectorID'] = omemd.instrument(series).Detector.get_ID()
    metadata['DetectorModel'] = omemd.instrument(series).Detector.get_Type()
    metadata['ObjNA'] = omemd.instrument(series).Objective.get_LensNA()
    metadata['ObjID'] = omemd.instrument(series).Objective.get_ID()
    metadata['ObjMag'] = omemd.instrument(series).Objective.get_NominalMagnification()

    # get channel names
    for c in range(metadata['SizeC']):
        metadata['Channels'].append(omemd.image(series).Pixels.Channel(c).Name)

    return metadata


def get_imgtype(imagefile):

    imgtype = None

    if imagefile.lower().endswith('.ome.tiff') or imagefile.lower().endswith('.ome.tif'):
        imgtype = 'ometiff'

    if imagefile.lower().endswith('.czi'):
        imgtype = 'czi'

    return imgtype


def get_metadata(imagefile, series=0):

    imgtype = get_imgtype(imagefile)
    print('Image Type: ', imgtype)
    md = None

    if imgtype == 'ometiff':
        with tifffile.TiffFile(imagefile) as tif:
            omexml = tif[0].image_description.decode("utf-8")

        print('Getting OME-TIFF Metadata ...')
        omemd = omexmlClass.OMEXML(omexml)
        md = get_metadata_ometiff(imagefile, omemd, series=series)

    if imgtype == 'czi':

        print('Getting CZI Metadata ...')
        md = get_metadata_czi(imagefile, dim2none=False)

    return md


def create_ipyviewer(array5d, metadata):

    t = widgets.IntSlider(description='T:',
                          min=1,
                          max=metadata['SizeT'],
                          step=1,
                          value=1,
                          continuous_update=False)

    z = widgets.IntSlider(description='Z:',
                          min=1,
                          max=metadata['SizeZ'],
                          step=1,
                          value=1,
                          continuous_update=False)

    c = widgets.IntSlider(description='C:',
                          min=1,
                          max=metadata['SizeC'],
                          step=1,
                          value=1)

    r = widgets.IntRangeSlider(description='Display Range:',
                               min=array5d.min(),
                               max=array5d.max(),
                               step=1,
                               value=[array5d.min(), array5d.max()],
                               continuous_update=False)

    # disable slider that are not needed
    if metadata['SizeT'] == 1:
        t.disabled = True
    if metadata['SizeZ'] == 1:
        z.disabled = True
    if metadata['SizeC'] == 1:
        c.disabled = True

    ui = widgets.VBox([t, z, c, r])

    def get_TZC(t, z, c, r):
        display_image(array5d, t=t, z=z, c=c, vmin=r[0], vmax=r[1])

    out = widgets.interactive_output(get_TZC, {'t': t, 'z': z, 'c': c, 'r': r})

    return out, ui  # , t, z, c, r


def display_image(array5d, t=0, c=0, z=0, vmin=0, vmax=1000):
    image = array5d[t - 1, z - 1, c - 1, :, :]
    # display the labelled image
    fig, ax = plt.subplots(figsize=(10, 10))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(image, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cm.gray)
    fig.colorbar(im, cax=cax, orientation='vertical')
    print('Min-Max (Current Plane):', image.min(), '-', image.max())


def get_TZC(t, z, c, r):

    display_image(array5d, t=t, z=z, c=c, vmin=r[0], vmax=r[1])


def get_metadata_czi(filename, dim2none=False):

    # get CZI object and read array
    czi = zis.CziFile(filename)
    mdczi = czi.metadata()

    # parse the XML into a dictionary
    metadatadict_czi = xmltodict.parse(mdczi)
    metadata = create_metadata_dict()

    # get directory and filename etc.
    metadata['Directory'] = os.path.dirname(filename)
    metadata['Filename'] = os.path.basename(filename)

    # add axes and shape information
    metadata['Axes'] = czi.axes
    metadata['Shape'] = czi.shape

    """
    metadata['Experiment'] = metadatadict_czi['ImageDocument']['Metadata']['Experiment']

    try:
        metadata['Experiment'] = metadatadict_czi['ImageDocument']['Metadata']['Experiment']
    except:
        metadata['Experiment'] = None

    try:
        metadata['HardwareSetting'] = metadatadict_czi['ImageDocument']['Metadata']['HardwareSetting']
    except:
        metadata['HardwareSetting'] = None

    try:
        metadata['CustomAttributes'] = metadatadict_czi['ImageDocument']['Metadata']['CustomAttributes']
    except:
        metadata['CustomAttributes'] = None
    """

    metadata['Information'] = metadatadict_czi['ImageDocument']['Metadata']['Information']
    metadata['PixelType'] = metadata['Information']['Image']['PixelType']
    metadata['SizeX'] = np.int(metadata['Information']['Image']['SizeX'])
    metadata['SizeY'] = np.int(metadata['Information']['Image']['SizeY'])

    try:
        metadata['SizeZ'] = np.int(metadata['Information']['Image']['SizeZ'])
    except:
        if dim2none:
            metadata['SizeZ'] = None
        if not dim2none:
            metadata['SizeZ'] = 1

    try:
        metadata['SizeC'] = np.int(metadata['Information']['Image']['SizeC'])
    except:
        if dim2none:
            metadata['SizeC'] = None
        if not dim2none:
            metadata['SizeC'] = 1

    channels = []
    for ch in range(metadata['SizeC']):
        try:
            channels.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                            ['Channels']['Channel'][ch]['ShortName'])
        except:
            try:
                channels.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                                ['Channels']['Channel']['ShortName'])
            except:
                channels.append(str(ch))

    metadata['Channels'] = channels

    try:
        metadata['SizeT'] = np.int(metadata['Information']['Image']['SizeT'])
    except:
        if dim2none:
            metadata['SizeT'] = None
        if not dim2none:
            metadata['SizeT'] = 1

    try:
        metadata['SizeM'] = np.int(metadata['Information']['Image']['SizeM'])
    except:
        if dim2none:
            metadatada['SizeM'] = None
        if not dim2none:
            metadata['SizeM'] = 1

    try:
        metadata['SizeB'] = np.int(metadata['Information']['Image']['SizeB'])
    except:
        if dim2none:
            metadatada['SizeB'] = None
        if not dim2none:
            metadata['SizeB'] = 1

    try:
        metadata['SizeS'] = np.int(metadata['Information']['Image']['SizeS'])
    except:
        if dim2none:
            metadatada['SizeS'] = None
        if not dim2none:
            metadata['SizeS'] = 1

    try:
        metadata['Scaling'] = metadatadict_czi['ImageDocument']['Metadata']['Scaling']
        metadata['XScale'] = float(metadata['Scaling']['Items']['Distance'][0]['Value']) * 1000000
        metadata['YScale'] = float(metadata['Scaling']['Items']['Distance'][1]['Value']) * 1000000
        metadata['XScaleUnit'] = metadata['Scaling']['Items']['Distance'][0]['DefaultUnitFormat']
        metadata['YScaleUnit'] = metadata['Scaling']['Items']['Distance'][1]['DefaultUnitFormat']
        try:
            metadata['ZScale'] = float(metadata['Scaling']['Items']['Distance'][2]['Value']) * 1000000
            metadata['ZScaleUnit'] = metadata['Scaling']['Items']['Distance'][2]['DefaultUnitFormat']
        except:
            if dim2none:
                metadata['ZScale'] = None
            if not dim2none:
                metadata['ZScale'] = 1.0
    except:
        metadata['Scaling'] = None

    """
    try:
        metadata['DisplaySetting'] = metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
    except:
        metadata['DisplaySetting'] = None

    try:
        metadata['Layers'] = metadatadict_czi['ImageDocument']['Metadata']['Layers']
    except:
        metadata['Layers'] = None
    """

    # try to get software version
    metadata['SW-Name'] = metadata['Information']['Application']['Name']
    metadata['SW-Name'] = metadata['Information']['Application']['Version']
    metadata['AcqDate'] = metadata['Information']['Image']['AcquisitionDateAndTime']

    metadata['Instrument'] = metadata['Information']['Instrument']

    # get objective data
    try:
        metadata['ObjName'] = metadata['Instrument']['Objectives']['Objective']['@Name']
    except:
        metadata['ObjName'] = None

    try:
        metadata['ObjImmersion'] = metadata['Instrument']['Objectives']['Objective']['Immersion']
    except:
        metadata['ObjImmersion'] = None

    try:
        metadata['ObjNA'] = np.float(metadata['Instrument']['Objectives']['Objective']['LensNA'])
    except:
        metadata['ObjNA'] = None

    try:
        metadata['ObjID'] = metadata['Instrument']['Objectives']['Objective']['@Id']
    except:
        metadata['ObjID'] = None

    try:
        metadata['TubelensMag'] = np.float(metadata['Instrument']['TubeLenses']['TubeLens']['Magnification'])
    except:
        metadata['TubelensMag'] = None

    try:
        metadata['ObjNominalMag'] = np.float(metadata['Instrument']['Objectives']['Objective']['NominalMagnification'])
    except:
        metadata['ObjNominalMag'] = None

    try:
        metadata['ObjMag'] = metadata['ObjNominalMag'] * metadata['TubelensMag']
    except:
        metadata['ObjMag'] = None

    # get detector information
    try:
        metadata['DetectorID'] = metadata['Instrument']['Detectors']['Detector']['@Id']
    except:
        metadata['DetectorID'] = None

    try:
        metadata['DetectorModel'] = metadata['Instrument']['Detectors']['Detector']['@Name']
    except:
        metadata['DetectorModel'] = None

    try:
        metadata['DetectorName'] = metadata['Instrument']['Detectors']['Detector']['Manufacturer']['Model']
    except:
        metadata['DetectorName'] =  None


    # delete some key from dict
    del metadata['Instrument']
    del metadata['Information']
    del metadata['Scaling']

    # close CZI file
    czi.close()

    return metadata


def get_array_czi(filename, replacezero=True):

    # get CZI object and read array
    czi = zis.CziFile(filename)
    cziarray = czi.asarray()

    if replacezero:
        array = replaceZeroNaN(array, value=0)

    czi.close()

    return cziarray


def replaceZeroNaN(data, value=0):

    data = data.astype('float')
    data[data == value] = np.nan

    return data
