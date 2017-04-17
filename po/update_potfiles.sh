#!/bin/sh
# Builds po/POTFILES.in and po/tmp/*.h for intltool-update to chew on.

set -e

TMPFILE=tmp/POTFILES.in.unsorted
OUTFILE=POTFILES.in

mkdir -p "tmp"

# List Python code that imports either the standard gettext module, or
# our GLib-based compat mocules that supports C_(). Intltool-update
# knows about python code using syntax like the C macros.

git grep --full-name --files-with-matches "^from gettext import" .. \
    >"$TMPFILE"
git grep --full-name --files-with-matches "^from lib.gettext import" .. \
    >>"$TMPFILE"

# Builder XML and resource definitions are converted to a .h file in
# po/tmp which intltool-update can then work on.

for ui_file in ../gui/resources.xml ../gui/*.glade; do
    echo "Extracting strings from $ui_file..."
    intltool-extract --type=gettext/glade "$ui_file"
    tmp_h=`basename "$ui_file"`.h
    if ! test -f "tmp/$tmp_h"; then
        echo >&2 "warning: intltool-extract did not create tmp/$tmp_h"
        continue
    fi
    echo >>"$TMPFILE" "po/tmp/$tmp_h"
done

# Sort the output file for greater diffability.

sort "$TMPFILE" > "$OUTFILE"


# Update mypaint.pot too.
# This is committed to allow users on WebLate to begin translations for
# new languages.

intltool-update --verbose --gettext-package mypaint --pot

