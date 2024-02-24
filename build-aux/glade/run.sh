#!/bin/sh
# Runs the Glade UI designer with a local catalog, for design and development
# purposes. No need to install the catalog files.

dir="`dirname $0`"
GLADE_CATALOG_SEARCH_PATH="$dir"
if test "x$XDG_DATA_DIRS" = "x"; then
    XDG_DATA_DIRS="/usr/local/share/:/usr/share/"
fi
XDG_DATA_DIRS="${dir}:${XDG_DATA_DIRS}"

export GLADE_CATALOG_SEARCH_PATH
export XDG_DATA_DIRS
exec glade "$@"

