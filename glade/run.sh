#!/bin/sh
# Runs the Glade UI designer with a local catalog, for design and development
# purposes. No need to install the catalog files.

TOPDIR="`dirname $0`/.."
GLADE_DEFS_FILE="mypaint.glade"
GLADE_CATALOG_PATH="glade"
XDG_DATA_DIRS="$GLADE_CATALOG_PATH:$XDG_DATA_DIRS"

cd "$TOPDIR"
export GLADE_CATALOG_PATH
export XDG_DATA_DIRS
if test -f $GLADE_DEFS_FILE; then
    exec glade $GLADE_DEFS_FILE
else
    echo "Create $GLADE_DEFS_FILE and I'll load it for you next time."
    exec glade
fi

