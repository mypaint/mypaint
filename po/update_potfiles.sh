#!/usr/bin/env bash

# Generate the po template file from the sources

# Python files containing "from [lib.]gettext import ..."
ggrep="git grep --full-name --files-with-matches"
py_files=$($ggrep "^from .*gettext import" .. | grep "\.py$" | sed 's:^:../:')

# UI resource and layout files
ui_files=$($ggrep "^<interface>" .. | grep -v "\.py$" | sed 's:^:../:')

# Temp files
py_messages=$(mktemp)
ui_messages=$(mktemp)
tmp_messages=$(mktemp)

# Extract python strings, w. support for the pgettext alias we use: C_(ctxt, msg)
xgettext -F -c -o $py_messages -kC_:1c,2 $py_files
# Extract ui strings from the xml files
xgettext -F -c -o $ui_messages -LGlade $ui_files

# Concatenate to a single template file, stripping out info
# about the different origins and cleaning up the header string
# (somewhat brittle due to static line numbers, should be improved).
msgcat -t UTF-8 -o - $py_messages $ui_messages | sed -E '/#\..*#-#-#-#.*/ d' > $tmp_messages
cat <(head -n 20 $py_messages) <(tail -n+33 $tmp_messages) > mypaint.pot
