#!/usr/bin/env python3
# Extracts symbolic icons from a contact sheet using Inkscape.
# Copyright (c) 2013 Andrew Chadwick <a.t.chadwick@gmail.com>
# Copyright (c) 2020 The MyPaint Team
#
# Originally based on Jakub Steiner's r.rb,
# rewritten in Python for the MyPaint distrib.
# Jakub Steiner <jimmac@gmail.com>

# Depends on python-scour

## Imports
from __future__ import division, print_function

import argparse
from copy import deepcopy
import gzip
import logging
import os
import sys
import xml.etree.ElementTree as ET

from scour.scour import start as scour_optimize

logger = logging.getLogger(__file__)

# Constants

ICON_SHEET = "symbolic-icons.svgz"
OUTPUT_ICONS_ROOT = "../desktop/icons"
OUTPUT_THEME = "hicolor"
XLINK = "http://www.w3.org/1999/xlink"
SVG = "http://www.w3.org/2000/svg"
NAMESPACES = {
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
    "svg": SVG,
    "xlink": XLINK,
}
SUFFIX24 = ":24"


# Utilities

class FakeFile:
    """ String wrapper providing a subset of the file interface
    Used for the call to scour's optimizer for both input and output.
    """

    def __init__(self, string):
        self.string = string

    def write(self, newstring):
        self.string = newstring

    def read(self, ):
        return self.string

    def close(self):
        pass

    def name(self):
        return ""


class FakeOptions:
    """ Stand-in for values normally returned from an OptionParser
    Used for the call to scour's optimizer
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def get_by_attrib(parent, pred):
    """Utility function for retrieveing child elements by attribute predicates
    """
    return [c for c in parent.getchildren() if pred(c.attrib)]


def by_attrib(parent, attr, attr_pred):
    """Retrieves the elements matching an attribute predicate
    The predicate is only evaluated if the attribute is present.
    """
    def pred(attribs):
        if attr in attribs:
            return attr_pred(attribs[attr])
        else:
            return False
    return get_by_attrib(parent, pred)


# This reference resolution only handles simple cases
# and does not produce optimal (or even correct) results
# in the general case, but is sufficient for the current icons.
def resolve_references(refsvg, base):
    """Resolve external references in ``base``
    Any <use> tag in base will be replaced with the element it refers
    to unless that element is already a part of base. Assumes valid svg.
    """
    def deref(parent, child, index):
        if child.tag.endswith('use'):
            refid = gethref(child)
            el = base.find('.//*[@id="%s"]' % refid)
            if el is None:  # Need to fetch the reference
                ob = deepcopy(refsvg.find('.//*[@id="%s"]' % refid))
                assert ob is not None
                transform = child.get('transform')
                if transform:
                    ob.set('transform', transform)
                parent.remove(child)
                parent.insert(index, ob)
                deref(parent, ob, index)  # Recurse in case of use -> use
        else:
            for i, c in enumerate(child.getchildren()):
                deref(child, c, i)

    for i, c in enumerate(base.getchildren()):
        deref(base, c, i)


def extract_icon(svg, icon_elem, output_dir, output_svg):
    """Extract one icon"""
    group_id = icon_elem.attrib['id']
    logger.info("Extracting %s", group_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if group_id.endswith(SUFFIX24):
        size = 24
        icon_name = group_id[:-3] + "-symbolic"
    else:
        size = 16
        icon_name = group_id + "-symbolic"
    # Remove the backdrop (should always be the
    # first element, but we don't rely on it).
    backdrop, = by_attrib(
        icon_elem, '{%s}href' % XLINK,
        lambda s: s.startswith('#icon-base'))
    icon_elem.remove(backdrop)

    # Resolve references to objects outside of the icon
    resolve_references(svg, icon_elem)

    # Remove the `transform` attribute
    root = output_svg.getroot()
    icon_elem.attrib.pop('transform')
    for c in icon_elem.getchildren():
        root.append(c)
    root.set('width', str(size))
    root.set('height', str(size))

    # Remove unused style elements
    clean_styles(root)
    svgstr = FakeFile(ET.tostring(output_svg.getroot(), encoding="utf-8"))
    options = FakeOptions(
        newlines=False, shorten_ids=True, digits=5,
        strip_ids=True, remove_descriptive_elements=True,
        indent_depth=0, quiet=True
    )
    scour_optimize(options, svgstr, svgstr)
    output_file = os.path.join(output_dir, "%s.svg" % (icon_name,))
    with open(output_file, 'wb') as f:
        f.write(svgstr.read())


def parse_style(string):
    """Given a well-formed style string, return the corresponding dict"""
    return {k: v for k, v in
            (p.split(':') for p in string.split(';') if ':' in p)}


def serialize_style(style_dict):
    """Serialize a dict to a well-formed style string"""
    return ";".join([k + ":" + v for k, v in style_dict.items()])


def clean_styles(elem):
    """Recursively remove useless style attributes
    Also moves common fill declarations to the topmost element.
    """
    fill_cols = clean_style(elem)
    for child in elem:
        fill_cols = fill_cols.union(clean_styles(child))
    # Consolidate fill color (move it to the least common denominator)
    if len(fill_cols) == 1:  # Generally this should be the case
        for c in elem:
            # Exclude circles from the 'fill' attribute removal,
            # since gtk does handle the rendering correctly without it.
            if 'circle' not in c.tag and 'style' in c.attrib:
                style = parse_style(c.attrib['style'])
                style.pop('fill', None)
                if not style:
                    c.attrib.pop('style')
                else:
                    c.attrib['style'] = serialize_style(style)
        style = parse_style(elem.attrib.get('style', ''))
        style['fill'] = list(fill_cols)[0]
        elem.attrib['style'] = serialize_style(style)
    return fill_cols


def gethref(icon):
    return icon.attrib['{%s}href' % XLINK][1:]


def clean_style(elem):
    """Remove unused style attributes based on a fixed list
    A style attribute is removed if:
    1. it is not in the list
    2. it is in the list, but with a default value
    Expand the list as is necessary.
    """
    # Remove unused stuff from <use> elements - split out
    if elem.tag.endswith('use'):
        keys = list(elem.attrib.keys())
        for k in keys:
            if k in {'width', 'height', 'x', 'y'}:
                elem.attrib.pop(k)
    # At the point this is run, it is assumed that all strokes
    # have been converted to paths, hence the only values we
    # care about are opacity and fill. This may change in the
    # future, hence the more general implementation.
    key = 'style'
    useful = ["opacity", "fill"]
    defaults = {"opacity": (float, 1.0)}

    def is_default(k, v):
        if k in defaults:
            conv, default = defaults[k]
            return conv(v) == default
        return False
    styles = None
    if key in elem.attrib:
        styles = parse_style(elem.attrib[key])
        cleaned = {k: v for k, v in styles.items()
                   if k in useful and not is_default(k, v)}
        if cleaned:
            elem.attrib[key] = serialize_style(cleaned)
        else:
            elem.attrib.pop(key)
    # Return the fill color in a singleton (or empty) set
    return {v for k, v in (styles or dict()).items() if k == 'fill'}


def is_icon(e):
    valid_tag = e.tag in {s % SVG for s in ('{%s}g', '{%s}use')}
    return valid_tag and e.get('id') and e.get('id').startswith('mypaint-')


def get_icon_layer(svg):
    return svg.find('svg:g[@id="icons"]', NAMESPACES)


def extract_icons(svg, basedir, *ids):
    """Extract icon groups using Inkscape, both 16px scalable & 24x24"""

    # Make a copy of the tree
    base = deepcopy(svg)
    baseroot = base.getroot()
    # Empty the copy - it will act as the base for each extracted icon
    for c in baseroot.getchildren():
        baseroot.remove(c)

    icon_layer = get_icon_layer(svg)
    icons = (i for i in icon_layer if is_icon(i))
    if ids:
        icons = (i for i in icons if i.get('id') in ids)
    num_extracted = 0
    for icon in icons:
        num_extracted += 1
        iconid = icon.get('id')
        if icon.tag.endswith('use'):
            icon = deepcopy(icon_layer.find('*[@id="%s"]' % gethref(icon)))
            icon.attrib['id'] = iconid
        typedir = "24x24" if iconid.endswith(SUFFIX24) else "scalable"
        outdir = os.path.join(basedir, typedir, "actions")
        extract_icon(svg, deepcopy(icon), outdir, deepcopy(base))
    logger.info("Finished extracting %d icons" % num_extracted)


def get_icon_ids(svg):
    """Returns the ids of elements marked as being icons"""
    return (e.get('id') for e in get_icon_layer(svg) if is_icon(e))


def invalid_ids(svg, ids):
    return ids.difference(ids.intersection(set(get_icon_ids(svg))))


def main(options):
    """Main function for the tool"""
    logging.basicConfig(level=logging.INFO)
    for prefix, uri in NAMESPACES.items():
        ET.register_namespace(prefix, uri)
    basedir = os.path.join(OUTPUT_ICONS_ROOT, OUTPUT_THEME)
    if not os.path.isdir(basedir):
        logger.error("No dir named %r", basedir)
        sys.exit(1)
    logger.info("Reading %r", ICON_SHEET)
    with gzip.open(ICON_SHEET, mode='rb') as svg_fp:
        svg = ET.parse(svg_fp)
    # List all icon ids in sheet
    if options.list_ids:
        for icon_id in get_icon_ids(svg):
            print(icon_id)
    # Extract all icons
    elif options.extract_all:
        extract_icons(svg, basedir)
    # Extract icons by ids
    else:
        invalid = invalid_ids(svg, set(options.extract))
        if invalid:
            logger.error("Icon ids not found in icon sheet:\n{ids}".format(
                ids="\n".join(sorted(invalid))
            ))
            logger.error("Extraction cancelled!")
            sys.exit(1)
        else:
            extract_icons(svg, basedir, *options.extract)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    group = argparser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", dest="list_ids",
                       help="Print all icon ids to stdout and quit.")
    group.add_argument("--extract-all", action="store_true",
                       help="Extract all icons in the icon sheet.")
    group.add_argument("--extract", type=str, nargs="+", metavar="ICON_ID",
                       help="Extract the icons with the given ids.")
    options = argparser.parse_args()
    main(options)
