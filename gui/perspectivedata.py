# -*- coding: utf-8 -*-
#
# This file is part of MyPaint.
# Copyright (C) 2017 by Grzegorz Wójcik <grzegorz.w.1597@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

""" File for storing data for perspective mode """

from gettext import gettext as _


def update_data(data, node_id, additional_params):
    """ update data returned from get_default_data
        insert additional_params into node with id == node_id
    """
    for node in data['nodes']:
        if node['id'] == node_id:
            for (key, value) in additional_params.items():
                node[key] = value
            return


def get_default_data(name):
    """ return default data for perspective """
    data = {}
    if name == 'point':
        data = {'nodes': [{
            'type': 'VP',
            'id': '1',
            'locked': True,
        }], 'visualizations': [{
            'type': 'point',
            'nodes': ['1']
        }], 'edges': [], 'root': "1"}
    elif name == 'space':
        data = {'nodes': [{
            'type': 'Space',
            'id': 'space',
            'is_UI': 1
        }, {
            'type': 'VP',
            'id': 'Forward',
            'name': _('Forward'),
            'direction': [0, 0, 1],
        }, {
            'type': 'VP',
            'id': 'Side',
            'name': _('Side'),
            'direction': [1, 0, 0],
        }, {
            'type': 'VP',
            'id': 'Up',
            'name': _('Up'),
            'direction': [0, 1, 0],
            'locked': True
        }, {
            'type': 'Group',
            'id': 'config',
            'name': _('Configuration'),
            'enabled': False
        }], 'visualizations': [{
            'type': 'point',
            'nodes': ['Forward']
        }, {
            'type': 'point',
            'nodes': ['Side']
        }, {
            'type': 'point',
            'nodes': ['Up']
        }, {
            'type': 'space',
            'nodes': ['space', 'config']
        }], 'edges': [{
            'src': 'space',
            'dst': 'Forward'
        }, {
            'src': 'space',
            'dst': 'Side'
        }, {
            'src': 'space',
            'dst': 'Up',
        }, {
            'src': 'space',
            'dst': 'config'
        }], 'root': 'space'}
    elif name == 'group':
        data = {'nodes': [{
            'type': 'Group',
            'id': 'group',
            'name': _('group')
        }], 'edges': [
        ], 'root': 'group'}
    elif name == 'horizon':
        data = {'nodes': [{
            'type': 'Plane',
            'id': 'horizon',
            'name': _('horizon'),
            'is_compute': True,
            'compute_fct': 'plane',
            'compute_params': []
        }], 'edges': [
        ], 'root': 'horizon'}
    elif name == 'perspective_projection':
        data = {
            'nodes': [{
                'type': 'Group',
                'id': 'Root',
                'name': _('Root')
            }, {
                'type': 'RectilinearProjection',
                'id': 'View',
                'name': _('View'),
            }, {
                'type': 'VP',
                'id': 'R',
                'name': _('Right'),
                'direction': [1, 0, 1],
                'locked': True
            }, {
                'type': 'VP',
                'id': 'L',
                'name': _('Left'),
                'direction': [-1, 0, 1],
                'locked': True
            }, {
                'type': 'VP',
                'id': 'C',
                'name': _('Center'),
                'direction': [0, 0, 1],
                'locked': True
            }, {
                'type': 'Plane',
                'id': 'implicit_plane',
                'name': 'implicit plane',
                'tag': '__implicit_plane',
                'is_compute': True,
                'compute_fct': 'plane',
                'compute_params': []
            }, {
                'type': 'Group',
                'id': 'SP_Config',
                'name': _('Configuration'),
                'enabled': False
            }], 'visualizations': [{
                'type': 'projection',
                'nodes': ['View', 'SP_Config']
            }, {
                'type': 'point',
                'nodes': ['R']
            }, {
                'type': 'point',
                'nodes': ['L']
            }, {
                'type': 'point',
                'nodes': ['C']
            }], 'edges': [{
                'src': 'Root',
                'dst': 'View'
            }, {
                'src': 'View',
                'dst': 'R'
            }, {
                'src': 'View',
                'dst': 'L'
            }, {
                'src': 'View',
                'dst': 'C'
            }, {
                'src': 'View',
                'dst': 'implicit_plane'
            }, {
                'src': 'R',
                'dst': 'View',
                'type': 'VIEW'
            }, {
                'src': 'L',
                'dst': 'View',
                'type': 'VIEW'
            }, {
                'src': 'C',
                'dst': 'View',
                'type': 'VIEW'
            }, {
                'src': 'implicit_plane',
                'dst': 'View',
                'type': 'VIEW'
            }, {
                'src': 'Root',
                'dst': 'SP_Config'
            }, {
                'src': 'SP_Config',
                'dst': 'View',
                'type': 'VIEW'
            }], 'root': 'Root'
        }
    else:
        raise Exception('unknown default data')
    return data
