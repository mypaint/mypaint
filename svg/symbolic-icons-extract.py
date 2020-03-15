#!/usr/bin/python3
# Extracts symbolic icons from a contact sheet using Inkscape.
# Copyright (c) 2013 Andrew Chadwick <a.t.chadwick@gmail.com>
# Copyright (c) 2020 The MyPaint Team
#
# Based on Jakub Steiner's r.rb, rewritten in Python for the MyPaint distrib.
# Jakub Steiner <jimmac@gmail.com>

# Depends on inkscape and python-scour

# usage: python symbolic-icons-extract.py [GROUP_ID(s)]

# In order to run correctly, this script requires Inkscape 0.92
# or earlier to be available in $PATH. The flags have changed
# for Inkscape 1.0, and currently it does not provide the functionality
# required (at least not correctly).
#
# To export all icons, the output from running this script without
# arguments can be piped via xargs to the script itself, like this:
#
# ./symbolic-icons-extract.py | xargs ./symbolic-icons-extract.py
#
# This is generally not recommended, however (it takes a long time).

## Imports
from __future__ import division, print_function

import os
import re
import sys
import xml.etree.ElementTree as ET
import logging
import subprocess
import gzip

logger = logging.getLogger(__name__)

## Constants

CONTACT_SHEET = "symbolic-icons.svgz"
OUTPUT_ICONS_ROOT = "../desktop/icons"
OUTPUT_THEME = "hicolor"
INKSCAPE = "inkscape"
SCOUR = "scour"
SCOUR_OPTIONS = [
    "--remove-descriptive-elements",
    "--enable-id-stripping",
    "--shorten-ids",
    "--no-line-breaks",
]
NAMESPACES = {
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
    "svg": "http://www.w3.org/2000/svg",
    "xlink": "http://www.w3.org/1999/xlink",
}
SUFFIX24 = ":24"

# Regular expression to strip out 'color="#000000"'
replace_color = re.compile('[a-z-]*color="#000000"')
unnecessary_width_n_heights = re.compile('(width|height)="[^"]{3,}"')
double_whitespace = re.compile('[ ]{2,}')
REGEXP_SUBS = [
    (replace_color, ''),
    (double_whitespace, ' '),
]

def extract_icon(svg, group_id, output_dir):
    """Extract one icon"""
    assert group_id.startswith("mypaint-")
    logger.info("Extracting %s", group_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if group_id.endswith(SUFFIX24):
        size = 24
        icon_name = group_id[:-3] + "-symbolic"
    else:
        size = 16
        icon_name = group_id + "-symbolic"
    output_tmp = os.path.join(output_dir, ".%s.TMP.svg" % (icon_name,))
    svg.write(output_tmp)
    cmd = [INKSCAPE, "-f", output_tmp,   # '--without-gui' doesn't work...
           "--select", group_id,
           "--verb=FitCanvasToSelection",
           "--verb=EditInvertInAllLayers", "--verb=EditDelete",
           "--verb=EditSelectAll",
           "--verb=SelectionUnGroup", "--verb=StrokeToPath",
           "--verb=FileVacuum", "--verb=FileSave",
           "--verb=FileClose", "--verb=FileQuit"]
    subprocess.check_call(cmd)
    svg = ET.parse(output_tmp)
    groups = svg.findall(".//svg:g", NAMESPACES)
    assert groups is not None
    for group in groups:
        remove_rects_of_size(group, size)
    clean_styles(svg.getroot())
    svg.write(output_tmp)
    output_file = os.path.join(output_dir, "%s.svg" % (icon_name,))
    cmd = [SCOUR] + SCOUR_OPTIONS + [output_tmp]
    # Scour fails to remove the metadata in a single pass, so we
    # run it a second time (pass previous values to stdin).
    optim = subprocess.check_output(cmd, universal_newlines=True)
    optim = subprocess.check_output(
        cmd[:-1], input=optim, universal_newlines=True)
    for regexp, subst in REGEXP_SUBS:
        logger.info("Applying %r" % regexp)
        optim = regexp.sub(subst, optim)
    with open(output_file, 'w') as f:
        f.write(optim)
    os.unlink(output_tmp)
    logger.info("Wrote %s", output_file)


def remove_rects_of_size(group, size):
    """Removes the backdrop 16x16 or 24x24 rect from an icon's group"""
    for rect in group.findall("./svg:rect", NAMESPACES):
        rw = int(round(float(rect.get("width", 0))))
        rh = int(round(float(rect.get("height", 0))))
        if rw == size and rh == size:
            logger.info("Removing Backdrop")
            logger.debug("removing %r (is %dpx)", rect, size)
            group.remove(rect)
    for path in group.findall("./svg:path", NAMESPACES):
        delete_id = path.get("id")
        if delete_id.startswith("use"):
            logger.info("Removing Backdrop")
            group.remove(path)


def parse_style(string):
    """Given a well-formed style string, return the corresponding dict"""
    return {k: v for k, v in
            (p.split(':') for p in string.split(';') if ':' in p)}


def serialize_style(style_dict):
    """Serialize a dict to a well-formed style string"""
    return ";".join([k + ":" + v for k, v in style_dict.items()])


def clean_styles(elem):
    """Recursively remove useless style attributes"""
    clean_style(elem)
    for child in elem.iter():
        if child is not elem:
            clean_styles(child)


def clean_style(elem):
    """Remove unused style attributes based on a fixed list
    A style attribute is removed if:
    1. it is not in the list
    2. it is in the list, but with a default value
    Expand the list as is necessary.
    """
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
    if key in elem.attrib:
        styles = parse_style(elem.attrib[key])
        cleaned = {k: v for k, v in styles.items()
                   if k in useful and not is_default(k, v)}
        elem.attrib[key] = serialize_style(cleaned)


def extract_icons(svg, basedir, group_ids):
    """Extract icon groups using Inkscape, both 16px scalable & 24x24"""
    for group_id in group_ids:
        group = svg.find(".//svg:g[@id='%s']" % (group_id,), NAMESPACES)
        if group is None:
            logger.error("No group named %r", group_id)
        else:
            outdir = os.path.join(basedir, "scalable", "actions")
            extract_icon(svg, group_id, outdir)
        group_id_24 = group_id + SUFFIX24
        group = svg.find(".//svg:g[@id='%s']" % (group_id_24,), NAMESPACES)
        if group is None:
            logger.info("%r: no 24px variant (%r)", group_id, group_id_24)
        else:
            outdir = os.path.join(basedir, "24x24", "actions")
            extract_icon(svg, group_id_24, outdir)


def show_icon_groups(svg):
    """Print groups from the contact sheet which could be icons"""
    layers = svg.findall("svg:g[@inkscape:groupmode='layer']", NAMESPACES)
    for layer in layers:
        groups = layer.findall(".//svg:g", NAMESPACES)
        if groups is None:
            continue
        for group in layer.findall(".//svg:g", NAMESPACES):
            group_id = group.get("id")
            if group_id is None:
                continue
            if (group_id.startswith("mypaint-") and
                    not group_id.endswith(SUFFIX24)):
                print(group_id)


def main():
    """Main function for the tool"""
    logging.basicConfig(level=logging.INFO)
    for prefix, uri in NAMESPACES.items():
        ET.register_namespace(prefix, uri)
    basedir = os.path.join(OUTPUT_ICONS_ROOT, OUTPUT_THEME)
    if not os.path.isdir(basedir):
        logger.error("No dir named %r", basedir)
        sys.exit(1)
    logger.info("Reading %r", CONTACT_SHEET)
    with gzip.open(CONTACT_SHEET, mode='rb') as svg_fp:
        svg = ET.parse(svg_fp)
    group_ids = sys.argv[1:]
    if group_ids:
        logger.info("Attempting to extract %d icon(s)", len(group_ids))
        extract_icons(svg, basedir, group_ids)
    else:
        logger.info("Listing groups which (might) represent exportable icons "
                    "in %r", CONTACT_SHEET)
        show_icon_groups(svg)
    logger.info("Done")


if __name__ == '__main__':
    main()
