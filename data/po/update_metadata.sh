#!/bin/sh
# Fix up header comments and other metadata.
# Header comments will only be replaced if they
# consist of an intact unchanged default block,
# and the language name can be grabbed from the file.

git_commit_year() {
    git show -s --format='%ai' "$1" | cut --delimiter='-' --fields=1
}

git_commit_year_range() {
    # commit log order, hash1 should be newer than hash2
    commit_hash1="$1"
    commit_hash2="$2"
    years=$(git_commit_year "$commit_hash1")
    if [ -n "$commit_hash2" ]; then
	year2=$(git_commit_year "$commit_hash2")
	if [ "$years" != "$year2" ]; then
	    years="$year2-$years"
	fi
    fi
    echo "$years"
}

replace_default_header() {
    # Replace the default blurb, if present
    # (don't indent the definition)
    default_blurb="# SOME DESCRIPTIVE TITLE.
# Copyright (C) YEAR THE PACKAGE'S COPYRIGHT HOLDER
# This file is distributed under the same license as the PACKAGE package.
# FIRST AUTHOR <EMAIL@ADDRESS>, YEAR."
    pofile="$1"
    if [ "$(head -n4 "$pofile")" = "$default_blurb" ]; then
	echo "Default blurb found!"

	# Grab the language team field
	# Sed command explanation: Grab string between ":" and "<", remove
	# leading/trailing spaces and trailing instances of '\n"'
	sed_commands='{s/^[^:]*: *([^<]*).*$/\1/;s/([ ]*$)|(\\n"$)//p}'
	lang=$(sed -E -n '/^"Language-Team:/'"$sed_commands" "$pofile" | tr -d '\n')
	if [ -z "$lang" -o "$lang" = "none" ]; then
	    echo "Language team not specified, skipping header generation!"
	    return 1
	fi
	title="$lang translation of MyPaint"

	# Grab year of last proper edit (assumes that weblate was used for it)
	grepstring="using Weblate"
	hash_newest=$(
	    git log --oneline "$pofile" |
		grep -m1 "$grepstring" |
		cut --delimiter=" " --fields=1)
	hash_oldest=$(
	    git log --reverse --oneline "$pofile" |
		head -n1 | cut --delimiter=" " --fields=1)
	year_range=$(git_commit_year_range "$hash_newest" "$hash_oldest")
	cr_line="Copyright (C) $year_range by The MyPaint Development Team"

	lc_line="This file is distributed under the same license as MyPaint."
	tmp=$(mktemp)
	{
	    echo "# $title"
	    echo "# $cr_line"
	    echo "# $lc_line"
	    tail -n +5 "$pofile"
	} > "$tmp" && mv "$tmp" "$pofile"
    fi
}

update_project_id() {
    # Update the "Project-Id-Version" field based on git tags
    pofile="$1"
    # Check when the file was last updated
    po_rev_date=$(sed -n -E '/^"PO-Rev/ s/[^0-9]*([^\\]*).*/\1/p' "$pofile")
    po_rev_commit=$(git rev-list -n1 --first-parent --before="$po_rev_date" master)
    git_tag=$(git describe --tags "$po_rev_commit")
    version_string="mypaint $git_tag"
    if ! grep -q ".*$version_string.*" "$pofile"; then
	echo "Updating Project-Id-Version ($version_string)"
	sed -i -E 's/^("Project-Id[^:]*:).*/\1 '"$version_string"'\\n"/' "$pofile"
    fi
}

update_files() {
    for pofile in "$@"; do
	echo "Processing $(basename "$pofile")"
	# If replace_default_header fails, there is no point
	# in setting the project id (file contains no translations)
	replace_default_header "$pofile" && update_project_id "$pofile"
    done
}

if [ -n "$1" ]; then
    update_files "$@"
else
    update_files ./*.po
fi
