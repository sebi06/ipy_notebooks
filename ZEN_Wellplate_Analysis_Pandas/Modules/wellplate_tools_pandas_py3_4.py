# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from ipy_progressbar import ProgressBar
import progressbar
from IPython.display import display
from ipywidgets import FloatProgress
import os


#WELLID_KEY = 'ImageSceneContainerName::Image Scene Container Name '
#ROWID_KEY = 'ImageSceneRow::Image Scene Row Index!!I'
#COLUMNID_KEY = 'ImageSceneColumn::Image Scene Column Index!!I'

WELLID_KEY = 'WellID'
ROWID_KEY = 'RowID'
COLUMNID_KEY = 'ColumnID'


def extract_labels(nr, nc):
    """
    Define helper function to be able to extract the well labels depending
    on the actual wellplate type. Currently supports 96, 384 and 1536 well plates.

    :param nr: number of rows of the wellplate, e.g. 8 (A-H) for a 96 wellplate
    :param nc: number of columns of the wellplate, e.g. 12 (1-12) for a 96 wellplate
    :return: lx, ly are list containing the actual row and columns IDs
    """

    # labeling schemes
    labelX = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
              '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
              '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36',
              '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', ]

    labelY = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
              'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF']

    lx = labelX[0:nc]
    ly = labelY[0:nr]

    return lx, ly


def convert_row_index(rowid):
    """
    This function converts the row index given as a letter (string) to its integer.
    Example: rowid as string, e.g. 'B'
    Output: rowIndex as integer, e.g. 2

    :param rowid - is a string representing the row a wellplate
    :return: rowindex - integer representing the row
    """

    rowids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
              'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF']

    # add one since the index in Python is zero-based
    # the column or row index from ZEN IAS is one-based
    rowindex = rowids.index(rowid) + 1

    return rowindex


def get_well_all_parameters(df, wellid, colname='all', wellID_key=WELLID_KEY, rowID_key=ROWID_KEY, colID_key=COLUMNID_KEY):
    """
    Gets all or specific columns for specific well.
    If the colname was specified, only this specific column will be returned.

    :param df: original dataframe
    :param wellid: wellID, e.g. 'B4'
    :param colname: name of column(s) from the original dataframe to be extracted
    :return: new_df - new dataframe containing only data for a specific well
    """

    #wellID_key = 'ImageSceneContainerName::Image Scene Container Name '
    wellID_key = WELLID_KEY
    new_df = df.loc[df[wellID_key] == wellid]

    if colname != 'all':
        new_df = new_df[[wellID_key, rowID_key, colID_key, colname]]

    return new_df


def get_well_row(df, rowid, rowID_key=ROWID_KEY,):
    """
    This function extracts all data based on the row index.

    :param df: original dataframe containing all wells
    :param rowid: index of row to be extracted
    :return: df_row - dataframe that only contains data for a specific row af a wellplate
    """

    rowindex = convert_row_index(rowid)
    df_row = df.loc[df[rowID_key] == rowindex]

    return df_row


# def create_heatmap(nr, nc, df, colname):
#     """
#     Creates an array representing the well plate. The array is initially filled with NaNs.
#     The values from dataframe will be filled in according th the rowIndex and columnIndex.
#     The output is numpy array.
#
#     :param nr: number of rows of the wellplate
#     :param nc: number of columns of the wellplate
#     :param df: dataframe
#     :param colname: name of colums from which the values will be transferred to the heatmap
#     :return: heatmap_df - is a dataframe representing the wellplate for the specified parameter
#
#     """
#
#     hm = np.full([nr, nc], np.nan)
#
#     for i in range(0, df.shape[0]):
#         # update well matrix at the correct position - indices have to be integers ...
#         hm[int(df['RowID'][i] - 1), int(df['ColumnID'][i] - 1)] = df[colname][i]
#
#     # get the labels for a 96 well plate and create a data frame from the numpy array
#     lx, ly = extract_labels(nr, nc)
#     heatmap_df = pd.DataFrame(hm, index=ly, columns=lx)
#
#     return heatmap_df


def convert_array_to_heatmap(hmarray, nr, nc):

    # get the labels for a well plate and create a data frame from the numpy array
    lx, ly = extract_labels(nr, nc)
    heatmap_dataframe = pd.DataFrame(hmarray, index=ly, columns=lx)

    return heatmap_dataframe


# def create_heatmap_from_dict(hmdict, paramkey):
#     """
#     Creates an array representing the well plate from heatmap dictionary.
#
#     :param hmdict:
#     :param paramkey:
#     :return: heatmap_df_key - a dataframe representing the wellplate for a certain parameter
#     """
#
#     hmarray = hmdict[paramkey]
#     nr = hmarray.shape[0]
#     nc = hmarray.shape[1]
#
#     # get the labels for a 96 well plate and create a data frame from the numpy array
#     lx, ly = extract_labels(nr, nc)
#     heatmap_df_key = pd.DataFrame(hmarray, index=ly, columns=lx)
#
#     return heatmap_df_key


def show_heatmap(heatmap_df, title, showvalues=False, plot=False):
    """
    This function shows a heatmap using the seaborn library: https://pypi.python.org/pypi/seaborn/

    :param heatmap_df - dataframe with shape of the well plate
    :param title - string used to the title of the plot
    :param showvalues - if True the values will be shown inside the heatmap
    :param plot - if True the plt.show() will be executed right away
    :return: ax - Axis with heatmap
    """

    sns.set()
    # configure the heatmap plot for the cell number depending on the well positions
    if showvalues == True:
        ax = sns.heatmap(heatmap_df, cmap="YlGnBu", linecolor='black', annot=True,
                         fmt='3.0f', annot_kws={"size": 10}, square=True)

    elif showvalues == False:
        ax = sns.heatmap(heatmap_df, cmap="YlGnBu", linecolor='black', square=True)

    # customize the plot to your needs
    ax.set_title(title, fontsize=16)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    if plot == True:
        plt.show()

    return ax


def rename_col_fromcsv_all(df):
    """
    Inside the ZEN image analysis software the columns of the result tables have
    have very long and inconvenient names. This function is designed to get rid
    of those names and replace them by shorter version.

    Important is to check for the order of the columns inside the CSV table.
    If this order is changes the renaming function must be adapted accordingly.

    :param df - dataframe containing mean features for all objects (AllRegions)
    :return: df: dataframe with shorter columns names and correct data types.
    """

    # delete unneeded rows
    df = df.drop([0])
    # rename the columns in order to use more descriptive names
    df.rename(columns={df.columns[0]: 'WellID',
                       df.columns[1]: 'RowID',
                       df.columns[2]: 'ColumnID',
                       df.columns[3]: 'PositionIndex',
                       df.columns[4]: 'CellNumber',
                       df.columns[5]: 'DAPI_MeanInt',
                       df.columns[6]: 'DAPI_MeanArea'}, inplace=True)

    # convert columns to numeric types when possible, since there might be issues
    # determine the correct data types for some columns
    df = df.convert_objects(convert_numeric=True)

    # convert columns to numeric types when possible --> convert to float
    df['DAPI_MeanInt'] = df['DAPI_MeanInt'].str.replace(',', '.').astype('float')
    df['DAPI_MeanArea'] = df['DAPI_MeanArea'].str.replace(',', '.').astype('float')

    return df


def rename_col_fromcsv_single(dfs, paramlist):
    """
    Inside the ZEN image analysis software the columns of the result tables have
    have very long and inconvenient names. This function is designed to get rid
    of those names and replace them by shorter version.

    Important is to check for the order of the columns inside the CSV table.
    If this order is changes the renaming function must be adapted accordingly.

    Convention for the ZEN Image Analysis Parameters order:

    WellID - RowID - ColumnID - ID - Index - Param1 - Param2 - ...

    WHEN THIS CONVENTION IS NOT USED, THE FOLLOWING CODE WILL NOT WORK AS EXPECTED.

    :param dfs - dataframe containing all features for single objects (SingleRegions)
    :param paramlist - list containing the desired column names (short version)
    :return: dfs - dataframe with shorter columns names and correct data types.
    """

    # delete unneeded rows --> contains the units
    dfs = dfs.drop([0])
    # rename columns
    dfs.rename(columns={dfs.columns[0]: 'WellID',
                        dfs.columns[1]: 'RowID',
                        dfs.columns[2]: 'ColumnID',
                        dfs.columns[3]: 'ID',
                        dfs.columns[4]: 'Index'}, inplace=True)

    for i in range(0, len(paramlist)):

        # rename the columns with measured parameters and correct types
        dfs.rename(columns={dfs.columns[i + 5]: paramlist[i]}, inplace=True)
        try:
            dfs[paramlist[i]] = dfs[paramlist[i]].str.replace(',', '.').astype('float')
        except:
            print('No correction of types possible for parameter: ', paramlist[i])

    # # convert columns to numeric types to avaoid problems with the data types and decimal separators
    # dfs['DAPI_Mean'] = dfs['DAPI_Mean'].str.replace(',', '.').astype('float')
    # dfs['Area'] = dfs['Area'].str.replace(',', '.').astype('float')
    # dfs['Perimeter'] = dfs['Perimeter'].str.replace(',', '.').astype('float')
    # dfs['Roundness'] = dfs['Roundness'].str.replace(',', '.').astype('float')

    return dfs


def create_heatmap_list_arrays(numparams, nr, nc):
    """
    This function creates a list of arrays depending on the number of
    measure parameters plus one additional array for the object number.
    The shape of the arrays depends on the well type.

    :param numparams - number of measured parameters w/o the actual object number
    :param nr - number of rows of wellplate
    :param nc - number of columns for wellplate
    :return: heatmaplist_array - list of arrays representing the measures parameters plus obne for the object number
    """

    heatmaplist_array = []

    for i in range(0, numparams + 1):
            # create list containing all heatmaps for number of objects + all measured parameters
        heatmaplist_array.append(np.full([nr, nc], np.nan))

    return heatmaplist_array


def fill_heatmaps(dfs, numparams, nr, nc, statfunc='mean', showbar=False, verbose=False, rowID_key=ROWID_KEY, colID_key=COLUMNID_KEY):
    """
    Create dictionary containing heatmaps (dataframes) for all measured parameters

    1) Determine how many wells actually contain data
    2) Loop over all wells
    3) Extract only data from current well from dataframe and calc statistics
    4) Save the results in a dictionary containing entries for all wells
    5) Fill the arrays with the values for the measured parameters from the well dictionary
    6) Create a dictionary that contains heatmaps (dataframes) for all measure parameters

    :param dfs - input data frame
    :param numparams - number of measured parameters except the object number
    :param nr - number of rows of well plate --> 96 plate = 8
    :param nc - number of rows of well plate --> 96 plate = 12
    :param statfunc - choice which statistics should be calculated
    :param verbose - if True more output will be shown
    :return: hm_dict - dictionary containing one dataframe for all measured parameters
    plus one entry for the heatmap containing the object numbers
    :return: welldata_dict - dictionary containing entries for every well analyzed
    with the values calculated by the statistical function
    """

    welldata_dict = {}
    heatmap_dict = {}

    # get all wells containing some data
    wellID_key = WELLID_KEY  # 'ImageSceneContainerName::Image Scene Container Name '
    print('---------------------------------------------------')
    print('wellID_key : ', wellID_key)
    print('Found keys:')
    print(dfs.keys())
    print('---------------------------------------------------')
    wells_real = dfs[wellID_key].value_counts()

    df_stats = pd.DataFrame(index=range(len(wells_real)), columns=dfs.columns)
    df_stats.drop(df_stats.columns[[3, 4]], axis=1, inplace=True)
    new_cols = df_stats.columns

    # create an additional columns of the object numbers
    df_obj = pd.DataFrame(index=range(len(wells_real)), columns=['ObjectNumbers'])

    if showbar == True:
        # initialize the progress bar
        #pb1 = ProgressBar(len(wells_real), title='Processing Wells')
        pb1 = iter(range(len(wells_real)))

        try:
            fp = FloatProgress(min=1,
                               max=len(wells_real),
                               step=1,
                               description='Processing Wells',
                               orientation='horizontal')
            display(fp)
        except:
            bar = progressbar.ProgressBar(redirect_stdout=True,
                                          max_value=len(wells_real))

    elif showbar == False:
        pb1 = iter(range(len(wells_real)))

    # iterate over all wells that were detected and do the statistics
    for well in pb1:

        try:
            fp.value += 1
        except:
            bar.update(well)

        # extract current dataframe for all existing wells
        current_wellid = wells_real.keys()[well]

        if verbose:
            print("Found data for wells : ",  current_wellid)

        # get all data for the current well from the over dataframe
        df_tmp = get_well_all_parameters(dfs, current_wellid)

        # fill in the wellID, rowID and colID into the new dataframe for the statistics
        df_stats.iloc[well][new_cols[0]] = current_wellid
        df_stats.iloc[well][new_cols[1]] = df_tmp.iloc[0][new_cols[1]]
        df_stats.iloc[well][new_cols[2]] = df_tmp.iloc[0][new_cols[2]]

        colnames = df_tmp.columns[list(range(5, 5 + numparams))]
        if statfunc == 'mean':
            stats_out = df_tmp.mean(axis=0)[colnames]
            for col in colnames:
                df_stats.iloc[well][col] = stats_out[col]
        elif statfunc == 'median':
            stats_out = df_tmp.median(axis=0)[colnames]
            for col in colnames:
                df_stats.iloc[well][col] = stats_out[col]
        elif statfunc == 'min':
            stats_out = df_tmp.min(axis=0)[colnames]
            for col in colnames:
                df_stats.iloc[well][col] = stats_out[col]
        elif statfunc == 'max':
            stats_out = df_tmp.max(axis=0)[colnames]
            for col in colnames:
                df_stats.iloc[well][col] = stats_out[col]

        # get number of entries and add them to stats data frame
        numobj_current_wellID = df_tmp.shape[0]
        # find the row index for the current wellID ...
        tmprow = df_stats[wellID_key].values.tolist().index(current_wellid)
        # ... and use the index to add the object number to the dataframe for the numbers
        df_obj['ObjectNumbers'][tmprow] = numobj_current_wellID

    # join the data frame with object numbers to df_stats
    df_stats = pd.concat([df_stats, df_obj], axis=1)

    # create welldata_dict
    for well in range(len(wells_real)):

        wellid = df_stats[wellID_key][well]
        # adding data to welldata_dict using the wellid)
        welldata_dict[wellid] = df_stats.iloc[well]

    for hm in range(3, df_stats.shape[1]):

        # create heatmap based on the platetype
        heatmap_array = np.full([nr, nc], np.nan)
        heatmap_name = df_stats.columns[int(hm)]

        # cycle to df_stats based on the columns nam and transfer data to heatmap
        for v in range(0, df_stats.shape[0]):

            rowindex = df_stats[rowID_key].iloc[v]
            colindex = df_stats[colID_key].iloc[v]
            hm_value = df_stats[heatmap_name].iloc[v]
            heatmap_array[int(rowindex) - 1, int(colindex) - 1] = hm_value

            # convert array to heatmap_dataframe
            heatmap_dict[heatmap_name] = convert_array_to_heatmap(heatmap_array, nr, nc)

    return heatmap_dict, welldata_dict


def showheatmap(heatmap, parameter2display, fontsize_title=16, fontsize_label=12,
                colormap='Blues', save=False, robust=True, filename='Test.czi'):

    # create figure with subplots
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # create the heatmap
    ax = sns.heatmap(heatmap, ax=ax,
                     cmap=colormap,
                     # linecolor='black',
                     # linewidths = 0.5,
                     square=True,
                     robust=robust,
                     annot=False,
                     cbar_kws={"shrink": 0.68})

    # customize the plot to your needs
    ax.set_title(parameter2display, fontsize=fontsize_title, fontweight='normal')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize_label)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize_label)

    # modify the labels of the colorbar
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=fontsize_label)

    if save:
        savename = filename[:-4] + '_HM_' + parameter2display + '.png'
        fig.savefig(savename, dpi=100, orientation='portrait', transparent=False, frameon=False)
        print('Heatmap image saved as: ', savename)
    else:
        savename = False

    return savename


def showheatmap_all(heatmap_dict, subplots, fontsize_title=16, fontsize_label=12,
                    colormap='Blues', save=False, robust=True, filename='Test.czi', deletelast=False):

    # create figure with subplots
    fig, axn = plt.subplots(subplots[0], subplots[1], figsize=(10, 8))
    # counter for the graphs to plot while iterating over all measured params
    plotid = -1

    # cycle heatmaps heatmaps
    # for key in heatmap_dict.iterkeys():  # Python 2
    for key in heatmap_dict.keys():  # python 3

        plotid = plotid + 1
        # get the desired heatmap from the dictionary containing all heatmaps
        heatmap_test = heatmap_dict[key]
        # create the actual heatmap
        ax = sns.heatmap(heatmap_test, ax=axn.flat[plotid],
                         cmap=colormap,
                         # linecolor='black',
                         # linewidths = 0.5,
                         square=True,
                         robust=robust,
                         annot=False,
                         cbar_kws={"shrink": 1.0})

        # customize the plot to your needs
        ax.set_title(key, fontsize=fontsize_title)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize_label)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize_label)

    # delete last subplot for an uneven number of parameters
    if deletelast:
        if np.mod(subplots[0] * subplots[1], 2) != 0:
            fig.delaxes(axn.flat[subplots[0] * subplots[1] - 1])

    if save:
        savename = filename[:-4] + '_HM_all.png'
        fig.savefig(savename, dpi=100, orientation='portrait', transparent=False, frameon=False)
        print('Heatmap image saved as: ', savename)
    else:
        savename = False

    return savename


def getwellIDfromfilename(filename):
    """
    This function has to be adapted depending the choosen filename:

    Example:
    ------------------------------------
    filename = nuc-B-04.tif

    colid:  04
    colindex: 4
    rowid: B
    rowindex: 2
    wellid: B2
    ------------------------------------
    :param filename: filename of the image where the name contains the wellinfo
    :return: wellid - well identifier, e.g. B4
    :return: rowindex - index of the row starting with 1
    :return: colindex - index of coulmn starting with 1
    """
    filename_base = os.path.basename(filename)
    filename_base_woext = os.path.splitext(filename_base)[0]
    # this function has to be adapted depending
    wellcoldigits = -2
    colid = filename_base_woext[wellcoldigits:]
    colindex = int(colid)
    rowid = filename_base_woext[(wellcoldigits - 2):(wellcoldigits - 1)]
    rowindex = convert_row_index(rowid)
    wellid = rowid + str(colindex)

    return wellid, rowindex, colindex


def addWellinfoColumns(dataframe):

    # add WellID, RowID and ColumnID to the existing dataframe
    dataframe.insert(0, 'WellID', 'A1')
    dataframe.insert(1, 'RowID', 1)
    dataframe.insert(2, 'ColumnID', 1)

    return dataframe


def wellinfo2dataframe(df, colname_with_info):
    """
    This function adds three additional columns to the beginning of the dataframe

    :param dataframe: input pandas dataframe
    :return: dataframe - modified dataframe
    """

    # add wellinfo to the first 3 columns
    df = addWellinfoColumns(df)

    for i in range(0, df.shape[0]):

        # get the well info based on the image filename for every row
        wellid, rowindex, colindex = getwellIDfromfilename(df[colname_with_info][i])
        # modify the dataframe accordingly
        df.set_value(i, 'WellID', wellid)
        df.set_value(i, 'RowID', rowindex)
        df.set_value(i, 'ColumnID', colindex)

    return df


def getrowandcolumn(platetype=96):
    """
    :param platetype - number total wells of plate (6, 24, 96, 384 or 1536)
    :return nr - number of rows of wellplate
    :return nc - number of columns for wellplate
    """
    platetype = int(platetype)

    if platetype == 6:
        nr = 2
        nc = 3
    elif platetype == 24:
        nr = 4
        nc = 6
    elif platetype == 96:
        nr = 8
        nc = 12
    elif platetype == 384:
        nr = 16
        nc = 24
    elif platetype == 1536:
        nr = 32
        nc = 48

    return nr, nc


def remove_units(df):

    # remove units from table
    df.drop([0], inplace=True)

    return df


def convert_dec_sep(df):

    nonmp = 5
    for i in range(nonmp, len(df.columns)):

        # rename the columns with measured parameters and correct types
        #df.rename(columns={df.columns[i]}, inplace=True)
        try:
            df.iloc[:, i] = df.iloc[:, i].str.replace(',', '.').astype('float')
        except:
            print('No correction of types possible for column: ', df.column[i + nonmp])

    return df
