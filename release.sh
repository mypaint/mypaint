#!/bin/sh
#: Make tarball releases from git, running tests. This does roughly
#: what 'make distcheck' would do if we were using autotools.
#:
#: Usage:
#:   $ ./release.sh [OPTIONS] [--] [OUTPUTDIR]
#:
#: Options:
#:   --help           show this message and exit ok
#:   --no-gitcheck    allow running with uncommitted changes
#:   --no-tests       avoid running the tests
#:   --no-cleanup     don't clean up the export location
#:   --headless       don't do anything requiring graphical output
#:   --debian-naming  outputs are debian-style .orig.tar.Xs (for PPAs)
#:   --simple-naming  outputs are just named mypaint.tar.X (builders)
#:   --git-naming     outputs are name with git commit number.
#:   --gzip-tarball   make the optional .tar.gz tarball
#:   --bzip2-tarball  make the optional .tar.bz2 tarball
#:
#: This script must be run from the top-level directory of a MyPaint
#: git checkout. By default, it makes just one .tar.xz output file
#: in the current working directory.
#:
#: Only skip tests and checks when debugging the export process itself.

set -e
set -x

SKIP_GITCHECK=false
SKIP_TESTS=false
SKIP_CLEANUP=false
DEBIAN_NAMING=false
SIMPLE_NAMING=false
GIT_NAMING=false
GZIP_TARBALL=false
BZIP2_TARBALL=false
OUTPUT_DIR="$(pwd)"

while test $# -gt 0; do
    case "$1" in
        --help)
            grep '^#:' $0
            exit 0;
            ;;
        --no-gitcheck)
            SKIP_GITCHECK=true
            shift
            ;;
        --no-tests)
            SKIP_TESTS=true
            shift
            ;;
        --no-cleanup)
            SKIP_CLEANUP=true
            shift
            ;;
        --debian-naming)
            DEBIAN_NAMING=true
            shift
            ;;
        --simple-naming)
            SIMPLE_NAMING=true
            shift
            ;;
        --git-naming)
            GIT_NAMING=true
            shift
            ;;
        --gzip-tarball)
            GZIP_TARBALL=true
            shift
            ;;
        --bzip2-tarball)
            BZIP2_TARBALL=true
            shift
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo >&2 "Unknown option $1 (try running with --help)"
            exit 1
            ;;
        *)
            break
            ;;
    esac
done
if test $# -gt 0; then
    OUTPUT_DIR="$1"
    shift
fi
if test $# -gt 0; then
    echo >&2 "Trailing junk in args: \"$@\" (try running with --help)"
    exit 1
fi


if ! test -d .git; then
    echo "Not at the root of a git repository"
    exit 1
fi
if ! $SKIP_GITCHECK; then
    if ! git diff --quiet; then
        echo "You have local changes, stage them first with 'git add'!"
        exit 1
    fi
fi

# Extract versions, either from the code and .git, or from release_info
# if it exists.

PYTHONPATH=. python2 lib/meta.py > "./.release_info.TMP"
. "./.release_info.TMP"

# Base version; a string like "1.1.0" for stable releases or "1.1.1-alpha"
# when making prereleases during the active development cycle.
base_version="$MYPAINT_VERSION_BASE"
formal_version="$MYPAINT_VERSION_FORMAL"
long_version="$MYPAINT_VERSION_CEREMONIAL"
echo "Base version: $base_version"

orig_dir="$(pwd)"

# Tarball naming
if $DEBIAN_NAMING; then
    debian_upstream_version=`echo $formal_version | sed -e 's/-/~/'`
    tarball_basename="mypaint_${debian_upstream_version}.orig.tar"
    exportdir_basename="mypaint-${debian_upstream_version}"
elif $SIMPLE_NAMING; then
    tarball_basename="mypaint.tar"
    exportdir_basename="mypaint"
elif $GIT_NAMING; then
    git_export_version=`echo $long_version | sed -e 's/gitexport/git/'`
    tarball_basename="mypaint-${git_export_version}.tar"
    exportdir_basename="mypaint-${git_export_version}"
else
    tarball_basename="mypaint-${formal_version}.tar"
    exportdir_basename="mypaint-${formal_version}"
fi

# Tarball version string is used for the directory
exportdir_location="/tmp/.mypaint-export-$$"
exportdir_path="$exportdir_location/$exportdir_basename"

# Construct release tmpdir
rm -rf "$exportdir_location"
mkdir -p "$exportdir_location"
git checkout-index -a -f --prefix="$exportdir_path/"

# Tidy up release tmpdir, and record info in it about what was used.  If
# the release_info file exists in a build tree, scons will write it into
# the generated ./mypaint script in place of information it would
# otherwise glean from .git.
cd "$exportdir_path"
rm -f release.sh
rm -fr .git*
cp -a "${orig_dir}/.release_info.TMP" "release_info"
rm -f "${orig_dir}/.release_info.TMP"
cd ..

# Create tarballs of release dir before we do any test builds
mkdir -p "$OUTPUT_DIR"
tarball="$OUTPUT_DIR/$tarball_basename"
rm -f "$tarball"
tar -cf "$tarball" "$exportdir_basename"

if $GZIP_TARBALL; then
    echo "Making $tarball.gz ..."
    gzip -9 --keep --force "$tarball"
fi

if $BZIP2_TARBALL; then
    echo "Making $tarball.bz2 ..."
    bzip2 -9 --keep --force "$tarball"
fi

echo "Making $tarball.xz ..."
xz -9 --force "$tarball"

# Build the release and test it
if $SKIP_TESTS; then
    echo "Skipping debug build and tests"
else
    echo "Making debug build inside $exportdir_path ..."
    # TODO: Probably need to update this part for setuptools.
    cd "$exportdir_path"
    python2 setup.py build
    echo "Running tests ..."
    python2 tests/test_mypaintlib.py
    python2 tests/test_compositeops.py
    python2 tests/test_rendering.py
    if ! $HEADLESS; then
        python2 tests/test_performance.py -a -c 1
        python2 tests/test_memory_leak.py -a -e
    fi
    echo "Done testing."
fi

# Clean up
cd "$orig_dir"
cat "$exportdir_path/release_info"
if $SKIP_CLEANUP; then
    echo "Skipping cleanup of $exportdir_location"
else
    echo "Cleaning up $exportdir_location ..."
    rm -fr "$exportdir_location"
fi

# Results
$GZIP_TARBALL && ls -sSh "$tarball".gz
$BZIP2_TARBALL && ls -sSh "$tarball".bz2
ls -sSh "$tarball".xz
echo "Done."
