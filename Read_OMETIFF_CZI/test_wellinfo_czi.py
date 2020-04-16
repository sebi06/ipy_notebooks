# -*- coding: utf-8 -*-
"""
@author: Sebi

File: test_wellinfo_czi.py
Date: 16.04.2019
Version. 0.2
"""

import imgfileutils as imf

filenames = [r'testdata/B4_B5_S=8_4Pos_perWell_T=2_Z=1_CH=1.czi',
             r'testdata/96well-SingleFile-Scene-05-A5-A5.czi',
             r'testdata/testwell96.czi',
             r'c:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Castor\testwell96.czi']

filename = filenames[3]


# get the metadata from the czi file
md = imf.get_metadata_czi(filename, dim2none=False)

# shape and dimension entry from CZI file as returned by czifile.py
print('CZI Array Shape : ', md['Shape'])
print('CZI Dimension Entry : ', md['Axes'])

# show dimensions
print('--------   Show Dimensions --------')
print('SizeS : ', md['SizeS'])
print('SizeT : ', md['SizeT'])
print('SizeZ : ', md['SizeZ'])
print('SizeC : ', md['SizeC'])

well2check = 'A5'
isids = imf.getImageSeriesIDforWell(md['Well_ArrayNames'], well2check)

print('WellList            : ', md['Well_ArrayNames'])
print('Well Column Indices : ', md['Well_ColId'])
print('Well Row Indices    : ', md['Well_ColId'])
print('WellCounter         : ', md['WellCounter'])
print('Different Wells     : ', md['NumWells'])
print('ImageSeries Ind. Well ', well2check, ' : ', isids)
