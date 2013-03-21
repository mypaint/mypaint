#!/bin/sh
# Runs the Glade UI designer with a local catalog, for design and development
# purposes. No need to install the catalog files.

GLADE_CATALOG_SEARCH_PATH="`dirname $0`"
XDG_DATA_DIRS="$GLADE_CATALOG_SEARCH_PATH:$XDG_DATA_DIRS"
export GLADE_CATALOG_SEARCH_PATH
export XDG_DATA_DIRS
exec glade "$@"

