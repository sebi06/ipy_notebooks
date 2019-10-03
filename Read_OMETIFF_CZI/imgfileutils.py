import czifile as zis
from apeer_ometiff_library import io, processing, omexmlClass
import os
import cziutils as czt
from skimage.external import tifffile
import ipywidgets as widgets
from matplotlib import pyplot as plt, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_metadata_dict():
    """
    A Python dictionary will be created to hold the relevant metadata.
    """

    metadata = {'Directory': '',
                'Filename': '',
                'Name': '',
                'AcqDate': '',
                'TotalSeries': 0,
                'SizeX': 0,
                'SizeY': 0,
                'SizeZ': 0,
                'SizeC': 0,
                'SizeT': 0,
                'DimOrder BF': 'n.a.',
                'NA': 0,
                'ObjMag': 0,
                'ObjID': 'n.a.',
                'XScale': 0,
                'YScale': 0,
                'ZScale': 0,
                'XScaleUnit': 'n.a.',
                'YScaleUnit': 'n.a.',
                'ZScaleUnit': 'n.a.',
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

    # get information baout the instrument and objective
    metadata['InstrumentID'] = omemd.instrument(series).get_ID()
    metadata['DetectorModel'] = omemd.instrument(series).Detector.get_Model()
    metadata['DetectorID'] = omemd.instrument(series).Detector.get_ID()
    metadata['DetectorModel'] = omemd.instrument(series).Detector.get_Type()
    metadata['NA'] = omemd.instrument(series).Objective.get_LensNA()
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
    metadata = None
    
    if imgtype == 'ometiff':
        with tifffile.TiffFile(imagefile) as tif:
            omexml = tif[0].image_description.decode("utf-8")
        
        print('Getting OME-TIFF Metadata ...')
        omemd = omexmlClass.OMEXML(omexml)
        metadata = get_metadata_ometiff(imagefile, omemd, series=series)
        
    return metadata


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
        t.disabled=True
    if metadata['SizeZ'] == 1:
        z.disabled=True
    if metadata['SizeC'] == 1:
        c.disabled=True
    
    ui = widgets.VBox([t, z, c, r])
    
    def get_TZC(t, z, c, r):
        display_image(array5d, t=t, z=z, c=c, vmin=r[0], vmax=r[1])
    
    out = widgets.interactive_output(get_TZC, { 't': t, 'z': z, 'c': c, 'r':r})
    
    return out, ui #, t, z, c, r


def display_image(array5d, t=0, c=0, z=0, vmin=0, vmax=1000):
    image = array5d[t-1, z-1, c-1, :, :]
    # display the labelled image
    fig, ax = plt.subplots(figsize=(10, 10))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(image, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cm.gray)
    fig.colorbar(im, cax=cax, orientation='vertical')
    print('Min-Max (Current Plane):', image.min(), '-', image.max())

def get_TZC(t, z, c, r):
    
    display_image(array5d, t=t, z=z, c=c, vmin=r[0], vmax=r[1])
        
        
        
