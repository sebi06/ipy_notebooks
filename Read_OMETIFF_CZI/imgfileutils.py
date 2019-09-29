import czifile as zis
from apeer_ometiff_library import io, processing, omexmlClass
import os
import cziutils as czt


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


def get_metadata_ometiff(filename, omexml, series=0):

    # create dictionary for metadata and get OME-XML data
    metadata = create_metadata_dict()
    md = omexmlClass.OMEXML(omexml)

    # get directory and filename etc.
    metadata['Directory'] = os.path.dirname(filename)
    metadata['Filename'] = os.path.basename(filename)
    metadata['AcqDate'] = md.image(series).AcquisitionDate
    metadata['Name'] = md.image(series).Name

    # get image dimensions
    metadata['SizeT'] = md.image(series).Pixels.SizeT
    metadata['SizeZ'] = md.image(series).Pixels.SizeZ
    metadata['SizeC'] = md.image(series).Pixels.SizeC
    metadata['SizeX'] = md.image(series).Pixels.SizeX
    metadata['SizeY'] = md.image(series).Pixels.SizeY
    # get number of series
    metadata['TotalSeries'] = md.get_image_count()

    # get dimension order
    metadata['DimOrder BF'] = md.image(series).Pixels.DimensionOrder

    # get the scaling
    metadata['XScale'] = md.image(series).Pixels.PhysicalSizeX
    metadata['XScaleUnit'] = md.image(series).Pixels.PhysicalSizeXUnit
    metadata['YScale'] = md.image(series).Pixels.PhysicalSizeY
    metadata['YScaleUnit'] = md.image(series).Pixels.PhysicalSizeYUnit
    metadata['ZScale'] = md.image(series).Pixels.PhysicalSizeZ
    metadata['ZScaleUnit'] = md.image(series).Pixels.PhysicalSizeZUnit

    # get all image IDs
    for i in range(md.get_image_count()):
        metadata['ImageIDs'].append(i)

    # get information baout the instrument and objective
    metadata['InstrumentID'] = md.instrument(series).get_ID()
    metadata['DetectorModel'] = md.instrument(series).Detector.get_Model()
    metadata['DetectorID'] = md.instrument(series).Detector.get_ID()
    metadata['DetectorModel'] = md.instrument(series).Detector.get_Type()
    metadata['NA'] = md.instrument(series).Objective.get_LensNA()
    metadata['ObjID'] = md.instrument(series).Objective.get_ID()
    metadata['ObjMag'] = md.instrument(series).Objective.get_NominalMagnification()

    # get channel names
    for c in range(metadata['SizeC']):
        metadata['Channels'].append(md.image(series).Pixels.Channel(c).Name)

    return metadata
