```python
########################################################################
# File       : CZI-ZARR Save Dask Array.ipynb
# Version    : 0.1
# Author     : czsrh
# Date       : 12.11.2019
# Insitution : Carl Zeiss Microscopy GmbH
#
# Disclaimer: Just for testing - Use at your own risk.
# Feedback or Improvements are welcome.
########################################################################
```

This notebook was mainly inspired by the following blogposts:

[Load Large Image Data with Dask Array](https://blog.dask.org/2019/06/20/load-image-data)

[Introducing napari: a fast n-dimensional image viewer in Python](https://ilovesymposia.com/2019/10/24/introducing-napari-a-fast-n-dimensional-image-viewer-in-python)


```python
# this can be used to switch on/off warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# import the libraries mentioned above
from apeer_ometiff_library import io, processing, omexmlClass
import czifile as zis
import xmltodict
import os
import time
import numpy as np
import ipywidgets as widgets
import napari
import imgfileutils as imf
import xml.etree.ElementTree as ET
import zarr
import dask
import dask.array as da
import glob
```


```python
# the directory contains 96 scenes of a wellplate as individual CZI files
# which where created by SplitScenesWriteFiles

# get list of all filenames
filenames = glob.glob(r'c:\Users\m1srh\Documents\Testdata_Zeiss\Castor\EMBL\96well\testwell96_Single_CZI\*.czi')

# show number of files
len(filenames)
```




    96




```python
def get_czi_array(filename):
    # get the array and the metadata
    array, metadata = imf.get_array_czi(filename)
    
    return array

metadata = imf.get_metadata_czi(filenames[0])
array_shape = metadata['Shape'][:-1]
array_dtype = metadata['NumPy.dtype']
print(array_shape)
print(array_dtype)

# find the indes for the Scenes dimensions from the dimstring
dims_dict, dimindex_list, numvalid_dims = imf.get_dimorder(metadata['Axes'])
dims_dict['S']

# lazy reading
lazy_arrays = [dask.delayed(get_czi_array)(fn) for fn in filenames]
lazy_arrays = [da.from_delayed(x, shape=array_shape, dtype=array_dtype) for x in lazy_arrays]
```

    (1, 1, 2, 1416, 1960)
    uint16
    


```python
# look at a singe array
lazy_arrays[0]
```




<table>
<tr>
<td>
<table>
  <thead>
    <tr><td> </td><th> Array </th><th> Chunk </th></tr>
  </thead>
  <tbody>
    <tr><th> Bytes </th><td> 11.10 MB </td> <td> 11.10 MB </td></tr>
    <tr><th> Shape </th><td> (1, 1, 2, 1416, 1960) </td> <td> (1, 1, 2, 1416, 1960) </td></tr>
    <tr><th> Count </th><td> 2 Tasks </td><td> 1 Chunks </td></tr>
    <tr><th> Type </th><td> uint16 </td><td> numpy.ndarray </td></tr>
  </tbody>
</table>
</td>
<td>
<svg width="374" height="151" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="25" y2="0" style="stroke-width:2" />
  <line x1="0" y1="25" x2="25" y2="25" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="25" style="stroke-width:2" />
  <line x1="25" y1="0" x2="25" y2="25" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.000000,0.000000 25.412617,0.000000 25.412617,25.412617 0.000000,25.412617" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="12.706308" y="45.412617" font-size="1.0rem" font-weight="100" text-anchor="middle" >1</text>
  <text x="45.412617" y="12.706308" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,45.412617,12.706308)">1</text>


  <!-- Horizontal lines -->
  <line x1="95" y1="0" x2="109" y2="14" style="stroke-width:2" />
  <line x1="95" y1="86" x2="109" y2="101" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="95" y1="0" x2="95" y2="86" style="stroke-width:2" />
  <line x1="109" y1="14" x2="109" y2="101" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="95.000000,0.000000 109.948598,14.948598 109.948598,101.642476 95.000000,86.693878" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Horizontal lines -->
  <line x1="95" y1="0" x2="215" y2="0" style="stroke-width:2" />
  <line x1="109" y1="14" x2="229" y2="14" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="95" y1="0" x2="109" y2="14" style="stroke-width:2" />
  <line x1="215" y1="0" x2="229" y2="14" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="95.000000,0.000000 215.000000,0.000000 229.948598,14.948598 109.948598,14.948598" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Horizontal lines -->
  <line x1="109" y1="14" x2="229" y2="14" style="stroke-width:2" />
  <line x1="109" y1="101" x2="229" y2="101" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="109" y1="14" x2="109" y2="101" style="stroke-width:2" />
  <line x1="229" y1="14" x2="229" y2="101" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="109.948598,14.948598 229.948598,14.948598 229.948598,101.642476 109.948598,101.642476" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="169.948598" y="121.642476" font-size="1.0rem" font-weight="100" text-anchor="middle" >1960</text>
  <text x="249.948598" y="58.295537" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,249.948598,58.295537)">1416</text>
  <text x="92.474299" y="114.168177" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(45,92.474299,114.168177)">2</text>
</svg>
</td>
</tr>
</table>




```python
# concatenate first n array
full_array = da.concatenate(lazy_arrays[:], axis=dims_dict['S'])
```


```python
# show full dask array
full_array
```




<table>
<tr>
<td>
<table>
  <thead>
    <tr><td> </td><th> Array </th><th> Chunk </th></tr>
  </thead>
  <tbody>
    <tr><th> Bytes </th><td> 1.07 GB </td> <td> 11.10 MB </td></tr>
    <tr><th> Shape </th><td> (1, 96, 2, 1416, 1960) </td> <td> (1, 1, 2, 1416, 1960) </td></tr>
    <tr><th> Count </th><td> 288 Tasks </td><td> 96 Chunks </td></tr>
    <tr><th> Type </th><td> uint16 </td><td> numpy.ndarray </td></tr>
  </tbody>
</table>
</td>
<td>
<svg width="392" height="151" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="34" y2="0" style="stroke-width:2" />
  <line x1="0" y1="25" x2="34" y2="25" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="25" style="stroke-width:2" />
  <line x1="0" y1="0" x2="0" y2="25" />
  <line x1="0" y1="0" x2="0" y2="25" />
  <line x1="1" y1="0" x2="1" y2="25" />
  <line x1="1" y1="0" x2="1" y2="25" />
  <line x1="1" y1="0" x2="1" y2="25" />
  <line x1="2" y1="0" x2="2" y2="25" />
  <line x1="2" y1="0" x2="2" y2="25" />
  <line x1="2" y1="0" x2="2" y2="25" />
  <line x1="3" y1="0" x2="3" y2="25" />
  <line x1="3" y1="0" x2="3" y2="25" />
  <line x1="3" y1="0" x2="3" y2="25" />
  <line x1="4" y1="0" x2="4" y2="25" />
  <line x1="4" y1="0" x2="4" y2="25" />
  <line x1="5" y1="0" x2="5" y2="25" />
  <line x1="5" y1="0" x2="5" y2="25" />
  <line x1="5" y1="0" x2="5" y2="25" />
  <line x1="6" y1="0" x2="6" y2="25" />
  <line x1="6" y1="0" x2="6" y2="25" />
  <line x1="6" y1="0" x2="6" y2="25" />
  <line x1="7" y1="0" x2="7" y2="25" />
  <line x1="7" y1="0" x2="7" y2="25" />
  <line x1="7" y1="0" x2="7" y2="25" />
  <line x1="8" y1="0" x2="8" y2="25" />
  <line x1="8" y1="0" x2="8" y2="25" />
  <line x1="8" y1="0" x2="8" y2="25" />
  <line x1="9" y1="0" x2="9" y2="25" />
  <line x1="9" y1="0" x2="9" y2="25" />
  <line x1="10" y1="0" x2="10" y2="25" />
  <line x1="10" y1="0" x2="10" y2="25" />
  <line x1="10" y1="0" x2="10" y2="25" />
  <line x1="11" y1="0" x2="11" y2="25" />
  <line x1="11" y1="0" x2="11" y2="25" />
  <line x1="11" y1="0" x2="11" y2="25" />
  <line x1="12" y1="0" x2="12" y2="25" />
  <line x1="12" y1="0" x2="12" y2="25" />
  <line x1="12" y1="0" x2="12" y2="25" />
  <line x1="13" y1="0" x2="13" y2="25" />
  <line x1="13" y1="0" x2="13" y2="25" />
  <line x1="13" y1="0" x2="13" y2="25" />
  <line x1="14" y1="0" x2="14" y2="25" />
  <line x1="14" y1="0" x2="14" y2="25" />
  <line x1="15" y1="0" x2="15" y2="25" />
  <line x1="15" y1="0" x2="15" y2="25" />
  <line x1="15" y1="0" x2="15" y2="25" />
  <line x1="16" y1="0" x2="16" y2="25" />
  <line x1="16" y1="0" x2="16" y2="25" />
  <line x1="16" y1="0" x2="16" y2="25" />
  <line x1="17" y1="0" x2="17" y2="25" />
  <line x1="17" y1="0" x2="17" y2="25" />
  <line x1="17" y1="0" x2="17" y2="25" />
  <line x1="18" y1="0" x2="18" y2="25" />
  <line x1="18" y1="0" x2="18" y2="25" />
  <line x1="18" y1="0" x2="18" y2="25" />
  <line x1="19" y1="0" x2="19" y2="25" />
  <line x1="19" y1="0" x2="19" y2="25" />
  <line x1="20" y1="0" x2="20" y2="25" />
  <line x1="20" y1="0" x2="20" y2="25" />
  <line x1="20" y1="0" x2="20" y2="25" />
  <line x1="21" y1="0" x2="21" y2="25" />
  <line x1="21" y1="0" x2="21" y2="25" />
  <line x1="21" y1="0" x2="21" y2="25" />
  <line x1="22" y1="0" x2="22" y2="25" />
  <line x1="22" y1="0" x2="22" y2="25" />
  <line x1="22" y1="0" x2="22" y2="25" />
  <line x1="23" y1="0" x2="23" y2="25" />
  <line x1="23" y1="0" x2="23" y2="25" />
  <line x1="23" y1="0" x2="23" y2="25" />
  <line x1="24" y1="0" x2="24" y2="25" />
  <line x1="24" y1="0" x2="24" y2="25" />
  <line x1="25" y1="0" x2="25" y2="25" />
  <line x1="25" y1="0" x2="25" y2="25" />
  <line x1="25" y1="0" x2="25" y2="25" />
  <line x1="26" y1="0" x2="26" y2="25" />
  <line x1="26" y1="0" x2="26" y2="25" />
  <line x1="26" y1="0" x2="26" y2="25" />
  <line x1="27" y1="0" x2="27" y2="25" />
  <line x1="27" y1="0" x2="27" y2="25" />
  <line x1="27" y1="0" x2="27" y2="25" />
  <line x1="28" y1="0" x2="28" y2="25" />
  <line x1="28" y1="0" x2="28" y2="25" />
  <line x1="29" y1="0" x2="29" y2="25" />
  <line x1="29" y1="0" x2="29" y2="25" />
  <line x1="29" y1="0" x2="29" y2="25" />
  <line x1="30" y1="0" x2="30" y2="25" />
  <line x1="30" y1="0" x2="30" y2="25" />
  <line x1="30" y1="0" x2="30" y2="25" />
  <line x1="31" y1="0" x2="31" y2="25" />
  <line x1="31" y1="0" x2="31" y2="25" />
  <line x1="31" y1="0" x2="31" y2="25" />
  <line x1="32" y1="0" x2="32" y2="25" />
  <line x1="32" y1="0" x2="32" y2="25" />
  <line x1="32" y1="0" x2="32" y2="25" />
  <line x1="33" y1="0" x2="33" y2="25" />
  <line x1="33" y1="0" x2="33" y2="25" />
  <line x1="34" y1="0" x2="34" y2="25" />
  <line x1="34" y1="0" x2="34" y2="25" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.000000,0.000000 34.374730,0.000000 34.374730,25.412617 0.000000,25.412617" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="17.187365" y="45.412617" font-size="1.0rem" font-weight="100" text-anchor="middle" >96</text>
  <text x="54.374730" y="12.706308" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,54.374730,12.706308)">1</text>


  <!-- Horizontal lines -->
  <line x1="104" y1="0" x2="118" y2="14" style="stroke-width:2" />
  <line x1="104" y1="86" x2="118" y2="101" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="104" y1="0" x2="104" y2="86" style="stroke-width:2" />
  <line x1="118" y1="14" x2="118" y2="101" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="104.000000,0.000000 118.948598,14.948598 118.948598,101.642476 104.000000,86.693878" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Horizontal lines -->
  <line x1="104" y1="0" x2="224" y2="0" style="stroke-width:2" />
  <line x1="118" y1="14" x2="238" y2="14" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="104" y1="0" x2="118" y2="14" style="stroke-width:2" />
  <line x1="224" y1="0" x2="238" y2="14" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="104.000000,0.000000 224.000000,0.000000 238.948598,14.948598 118.948598,14.948598" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Horizontal lines -->
  <line x1="118" y1="14" x2="238" y2="14" style="stroke-width:2" />
  <line x1="118" y1="101" x2="238" y2="101" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="118" y1="14" x2="118" y2="101" style="stroke-width:2" />
  <line x1="238" y1="14" x2="238" y2="101" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="118.948598,14.948598 238.948598,14.948598 238.948598,101.642476 118.948598,101.642476" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="178.948598" y="121.642476" font-size="1.0rem" font-weight="100" text-anchor="middle" >1960</text>
  <text x="258.948598" y="58.295537" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,258.948598,58.295537)">1416</text>
  <text x="101.474299" y="114.168177" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(45,101.474299,114.168177)">2</text>
</svg>
</td>
</tr>
</table>




```python
use_compression = False

# construct new filename for dask array
zarr_arrayname = os.path.join( os.path.dirname(filenames[0]), 'testwell96.zarr')
print('Saving ZARR Array to : ', zarr_arrayname)
                                               
# save to ZARR array if not already existing
if os.path.exists(zarr_arrayname):
    print('Dask Array already exits. Do not overwrite.')
if not os.path.exists(zarr_arrayname):
    # write data to disk using dask array
    if use_compression:
        from numcodecs import Blosc
        # save with compression
        full_array.to_zarr(zarr_arrayname, compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE))
    if not use_compression:
        # just use the "simple save" method
        full_array.to_zarr(zarr_arrayname)
```

    Saving ZARR Array to :  c:\Users\m1srh\Documents\Testdata_Zeiss\Castor\EMBL\96well\testwell96_Single_CZI\testwell96.zarr
    Dask Array already exits. Do not overwrite.
    


```python
# read image back from ZARR array
zarr_image = da.from_zarr(zarr_arrayname)

print('Array Type  : ', type(zarr_image))
print('Array Shape : ', zarr_image.shape)
```

    Array Type  :  <class 'dask.array.core.Array'>
    Array Shape :  (1, 96, 2, 1416, 1960)
    


```python
# switch to qt5 backend for napari viewer and wait a few seconds

%gui qt5
time.sleep(5)
```


```python
# initialize Napari Viewer and add the two channels as layes
viewer = napari.Viewer()
viewer.add_image(zarr_image[:, :, 0, :, :], name='A568', colormap='red', blending='additive')
viewer.add_image(zarr_image[:, :, 1, :, :], name='A488', colormap='green', blending='additive')
```




    <Image layer 'A488' at 0x248a1259ef0>



jupyter nbconvert CZI-ZARR Save Dask Array.ipynb --to slides --post serve
