#!/usr/bin/python
# Extracts symbolic icons from a contact sheet using Inkscape.
# Copyright (c) 2013 Andrew Chadwick <a.t.chadwick@gmail.com>
#
# Based on Jakub Steiner's r.rb, rewritten in Python for the MyPaint distrib.
# Jakub Steiner <jimmac@gmail.com>

# Depends on inkscape and python-scour

# usage: python symbolic-icons-extract.py [GROUP_ID(s)]

## Imports
from __future__ import division, print_function

import os
import sys
import xml.etree.ElementTree as ET
import logging
logger = logging.getLogger(__name__)
import subprocess
import gzip


## Constants

CONTACT_SHEET = "symbolic-icons.svgz"
OUTPUT_ICONS_ROOT = "../desktop/icons"
OUTPUT_THEME = "hicolor"
INKSCAPE = "inkscape"
SCOUR = "scour"
NAMESPACES = {
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
    "svg": "http://www.w3.org/2000/svg",
}
SUFFIX24 = ":24"


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
           "--verb=FileVacuum", "--verb=FileSave", "--verb=FileClose", "--verb=FileQuit"]
    subprocess.check_call(cmd)
    svg = ET.parse(output_tmp)
    groups = svg.findall(".//svg:g", NAMESPACES)
    assert groups is not None
    for group in groups:
        remove_rects_of_size(group, size)
    svg.write(output_tmp)
    output_file = os.path.join(output_dir, "%s.svg" % (icon_name,))
    cmd = [SCOUR, '--quiet', '-i', output_tmp, '-o', output_file]
    subprocess.check_call(cmd)
    os.unlink(output_tmp)
    logger.info("Wrote %s", output_file)


def remove_rects_of_size(group, size):
    """Removes the backdrop 16x16 or 24x24 rect from an icon's group"""
    for rect in group.findall("./svg:rect[@id='layer1']", NAMESPACES):
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
    for prefix, uri in NAMESPACES.iteritems():
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
