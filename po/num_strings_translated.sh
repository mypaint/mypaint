#!/bin/sh
#
# Print the number of translated strings to stdout
# (with the help of intltool-update) - polib fallback
# Fairly ugly; Any better shellscript solution using
# our existing dependencies would be appreciated.
#
# Input argument should be a locale code for an existing
# translation file, or omitted to get the string count total

loc_info()
{
    # Avoid version control clutter by using temp files
    cp "$1.po" "tmp_$1.po"
    # Redirect the info from stderr to stdout
    intltool-update -g mypaint --dist "tmp_$1" 2>&1
    rm "tmp_$1.po"
}

cd "$(dirname "$0")"

# Can't pipe directly from if-else
val=$(
if [ -z "$1" ]; then
    cp mypaint.pot mypaint.po
    loc_info mypaint | grep -o -E "[0-9]+ untranslated"
    rm -f mypaint.po
else
    loc_info "$1" | grep -o -E "[0-9]+ translated"
fi
)
echo "$val" | grep -o -E "[0-9]+"
