# -*- coding: utf-8 -*-
"""
@author: Sebi

File: test_wellinfo_czi.py
Date: 19.12.2019
Version. 0.1
"""

import imgfileutils as imf

filename = r'testdata/B4_B5_S=8_4Pos_perWell_T=2_Z=1_CH=1.czi'


md = imf.get_metadata_czi(filename, dim2none=False)

well2check = 'B4'
isids = imf.getImageSeriesIDforWell(md['Well_ArrayNames'], well2check)

print('WellList            : ', md['Well_ArrayNames'])
print('Well Column Indices : ', md['Well_ColId'])
print('Well Row Indices    : ', md['Well_ColId'])
print('WellCounter         : ', md['WellCounter'])
print('Different Wells     : ', md['NumWells'])
print('ImageSeries Ind. Well ', well2check, ' : ', isids)
