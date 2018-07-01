# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 11:01:21 2012

@author: M1SRH
"""

def parse_tracks_bfhws(exptree, verbose=False):

    # create track dictionary
    tracks = {}

    for node in exptree.findall('.//ExperimentBlocks/AcquisitionBlock/SubDimensionSetups/MultiTrackSetup/Track'):

        # create channel dictionary
        ch_attributes = {}
        # get channel name for current track
        chn = node.find('Channels/Channel').attrib.get('Name')
        # get channel activation status
        ch_attributes['isActivated'] = node.find('Channels/Channel').attrib.get('IsActivated')
        # get selection status
        ch_attributes['isSelected'] = node.find('Channels/Channel').attrib.get('IsSelected')
        # get name of BeforeHardwareSettings for current track
        bfhws_name = node.find('BeforeHardwareSetting').text
        if verbose:
            print('BFHWS Name : ', bfhws_name)
        # store BeforeHardwareSettings name in dictionary
        ch_attributes['BeforeHardwareSetting'] = bfhws_name
        tracks[chn] = ch_attributes

    return tracks


def parse_hardwaresettings(exptree, verbose=False):

    # create dictionary containing all keys for hardware settings
    bfhws = {}

    # parse all existing hardwaresettings
    for hwsname_node in exptree.findall('.//HardwareSettingsPool/HardwareSetting'):

        # get BeforeHardwareSetting name
        bfhwsname = hwsname_node.attrib.get('Name')
        # create dict for current BeforeHardwareSetting
        bfhws[bfhwsname] = {}
        if verbose:
            print('HardwareSetting Name: ', bfhwsname)
            print()

        components = {}

        for pcnode in exptree.findall('.//HardwareSettingsPool/HardwareSetting/ParameterCollection'):

            component_name = pcnode.attrib.get('Id')
            components[component_name] = {}
            if verbose:
                print(component_name)

            component_params = {}
            for child in pcnode.getchildren():

                parameter_name = child.tag
                for e in child.iter():
                    parameter_value = e.text
                    component_params[parameter_name] = parameter_value
                    if verbose:
                        print(parameter_name + ' : ' + parameter_value)
                    status = e.attrib.get('Status')
                    component_params['Status'] = status
                    isActivated = e.attrib.get('IsActivated')
                    component_params['IsActivated'] = isActivated

                components[component_name] = component_params
            #before_hws[bfhws] = component_params

        if verbose:
            print('*******************************************************************')

        bfhws[bfhwsname] = components

    return bfhws