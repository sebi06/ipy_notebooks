```python
########################################################################
# File       : using_apeer-ometiff-library.ipynb
# Version    : 0.1
# Author     : czsrh
# Date       : 17.12.2019
# Insitution : Carl Zeiss Microscopy GmbH
#
# Disclaimer: Just for testing - Use at your own risk.
# Feedback or Improvements are welcome.
########################################################################
```

***Reading OME-TIFF files from Python using the apeer-ometiff-library***

The APEER platform allows crating modules and workflows using basically any programming language due to its uderlying Docker(TM) technology. Nevertheless Python seens to be the favorite choice for mots of the time for various reasons:

* Simple enough to be used by resechers with scripting experience
* Powerful enough to create amazing computational tools
* A huge and very open community and the whole ecosystem behind it
* Probably the most popular langugae when it come to topics like Machine Learning

The topic or question what is the "best" image data format for microscopy is a very interesting and also quite difficult question. There are no easy answers and there is no right or wrong here.

Since the APEER platfrom tries to provide solutions our team decided that we must for sure support the currently most popular image data format for microscoscopy image data, which cleary is OME-TIFF (despite its known limitations). Therefore we explored "easy and simple" ways to read OME-TIFF for the most common use cases. We just want a simple python-based tool to read and write OME-TIFF without the need to include JAVA etc. into the modules. Therfore we reused parts of the existing python ecossystem, especially python-bioformats and tifffile, added some extra code and created a basic PyPi package.

This package can be easily inclued in every APEER module but obviousy it can be also used inside our python application or within jupyter notebook.

* [PyPi - apeer-ometiff-library](https://pypi.org/project/apeer-ometiff-library/)

* [PyPi - python-bioformats](https://pypi.org/project/python-bioformats/).

More information on the source code can be found on the APEER GitHub project page: [GitHub - apeer-ometiff-library](https://github.com/apeer-micro/apeer-ometiff-library)


```python
# import the libraries
from apeer_ometiff_library import io, processing, omexmlClass

# import script with some useful functions
import imgfileutils as imf
```


```python
# define your OME-TIFF file here
filename = r'c:\Temp\CellDivision_T=15_Z=20_CH=2_DCV.ome.tiff'

# extract XML and save it to disk
xmlometiff = imf.writexml_ometiff(filename)
```

    Created OME-XML file for testdata:  c:\Temp\CellDivision_T=15_Z=20_CH=2_DCV.ome.tiff
    

### Reading the OME-TIFF stack as an NumPy Array

The easily ready and OME-TIFF stack without the need to deal with the JAVA runtime the apeer-ometiff-library used the following code:
    
```python
def read_ometiff(input_path):
    with tifffile.TiffFile(input_path) as tif:
        array = tif.asarray()
        omexml_string = tif[0].image_description.decode('utf-8')

    # Turn Ome XML String to an Bioformats object for parsing
    metadata = omexmlClass.OMEXML(omexml_string)

    # Parse pixel sizes
    pixels = metadata.image(0).Pixels
    size_c = pixels.SizeC
    size_t = pixels.SizeT
    size_z = pixels.SizeZ
    size_x = pixels.SizeX
    size_y = pixels.SizeY

    # Expand image array to 5D of order (T, Z, C, X, Y)
    if size_c == 1:
        array = np.expand_dims(array, axis=-3)
    if size_z == 1:
        array = np.expand_dims(array, axis=-4)
    if size_t == 1:
        array = np.expand_dims(array, axis=-5)

    return array, omexml_string
```


```python
# Read metadata and array differently for OME-TIFF by using the io function of the apeer-ometiff library
 
# Return value is an array of order (T, Z, C, X, Y)
array, omexml = io.read_ometiff(filename)

# get the metadata for the OME-TIFF file
metadata, add_metadata = imf.get_metadata(filename)
```

    Image Type:  ometiff
    Getting OME-TIFF Metadata ...
    


```python
# check the shape of numpy array containing the pixel data
print('Array Shape: ', array.shape)

# get dimension order from metadata
print('Dimension Order (BioFormats) : ', metadata['DimOrder BF Array'])

# show dimensions and scaling
print('SizeT : ', metadata['SizeT'])
print('SizeZ : ', metadata['SizeZ'])
print('SizeC : ', metadata['SizeC'])
print('SizeX : ', metadata['SizeX'])
print('SizeY : ', metadata['SizeY'])
print('XScale: ', metadata['XScale'])
print('YScale: ', metadata['YScale'])
print('ZScale: ', metadata['ZScale'])
```

    Array Shape:  (2, 15, 20, 700, 700)
    Dimension Order (BioFormats) :  CTZYX
    SizeT :  15
    SizeZ :  20
    SizeC :  2
    SizeX :  700
    SizeY :  700
    XScale:  0.09057667
    YScale:  0.09057667
    ZScale:  0.32
    


```python
# show the complete metadata dictionary
for key, value in metadata.items():
    # print all key-value pairs for the dictionary
    print(key, ' : ', value)
```

    Directory  :  c:\Temp
    Filename  :  CellDivision_T=15_Z=20_CH=2_DCV.ome.tiff
    Extension  :  ome.tiff
    ImageType  :  ometiff
    Name  :  
    AcqDate  :  2016-02-12T09:41:02.4915604Z
    TotalSeries  :  1
    SizeX  :  700
    SizeY  :  700
    SizeZ  :  20
    SizeC  :  2
    SizeT  :  15
    Sizes BF  :  [1, 15, 20, 2, 700, 700]
    DimOrder BF  :  XYZTC
    DimOrder BF Array  :  CTZYX
    DimOrder CZI  :  None
    Axes  :  None
    Shape  :  None
    isRGB  :  None
    ObjNA  :  1.2
    ObjMag  :  50
    ObjID  :  Objective:1
    ObjName  :  None
    ObjImmersion  :  None
    XScale  :  0.09057667
    YScale  :  0.09057667
    ZScale  :  0.32
    XScaleUnit  :  None
    YScaleUnit  :  None
    ZScaleUnit  :  None
    DetectorModel  :  None
    DetectorName  :  []
    DetectorID  :  Detector:1
    InstrumentID  :  Instrument:1
    Channels  :  ['LED555', 'LED470']
    ImageIDs  :  [0]
    NumPy.dtype  :  None
    


```python
# Here we use https://ipywidgets.readthedocs.io/en/latest/ to create some simple and interactive controls to navigate
# through the planes of an multi-dimensional NumPy Array. 

# display data using ipywidgets
ui, out = imf.create_ipyviewer_ome_tiff(array, metadata)

# show the interactive widget
display(ui, out)
```


    Output()



    VBox(children=(IntSlider(value=1, description='Channel:', max=2, min=1), IntSlider(value=1, continuous_update=…


<img src="images\display_ometiff_ipywidgets.png" />


```python
# Here we use the Napari viewer (https://github.com/napari/napari) to visualize the complete OME-TIFF stack,
# which is represented by a multi-dimensional NumPy Array. 

# configure napari automatiaclly based on metadata and show the OME-TIFF stack
imf.show_napari(array, metadata)
```

    Initializing Napari Viewer ...
    Dim PosT :  1
    Dim PosC :  0
    Dim PosZ :  2
    Scale Factors :  [1.0, 1.0, 3.533, 1.0, 1.0]
    Shape Channel :  0 (15, 20, 700, 700)
    Scaling Factors:  [1.0, 1.0, 3.533, 1.0, 1.0]
    Scaling:  [0, 7941.0]
    Shape Channel :  1 (15, 20, 700, 700)
    Scaling Factors:  [1.0, 1.0, 3.533, 1.0, 1.0]
    Scaling:  [0, 33477.0]
    

<img src="images\display_ometiff_napari.png" />


```python
cd c:\Users\m1srh\Documents\GitHub\ipy_notebooks\Read_OMETIFF_CZI

jupyter nbconvert using_apeer-ometiff-library.ipynb --to slides --post serve

jupyter nbconvert using_apeer-ometiff-library.ipynb --to slides --post serve --SlidesExporter.reveal_theme=serif --SlidesExporter.reveal_scroll=True --SlidesExporter.reveal_transition=none
```
