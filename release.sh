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
#:   --gzip-tarball   make the optional .tar.gz tarball
#:   --bzip2-tarball  make the optional .tar.bz2 tarball
#:
#: This script must be run from the top-level directory of a MyPaint
#: git checkout. By default, it makes just one .tar.xz output file
#: in the current working directory.
#:
#: Only skip tests and checks when debugging the export process itself.

set -e

SKIP_GITCHECK=false
SKIP_TESTS=false
SKIP_CLEANUP=false
DEBIAN_NAMING=false
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

# Base version; a string like "1.1.0" for stable releases or "1.1.1-alpha"
# when making prereleases during the active development cycle.
base_version="`python lib/meta.py`"
echo "Version $base_version"

# Other version info we might want to capture in the tarball name or contents
rel_datestr=`date '+%Y%m%d'`
rel_gitrev=`git rev-parse --short HEAD`

# Prereleases have lengthier version strings generally, and especially long
# ones for the about box.
is_prerelease=false
formal_version="$base_version"
long_version="$base_version"
if echo $base_version | grep -q -- "-"; then
    is_prerelease=true
    formal_version="$base_version.$rel_datestr"
    long_version="$formal_version+gitexport.$rel_gitrev"
    # If somebody builds from a prerelease tarball rather than from a git
    # checkout, it is marked as "+gitexport" in the about box rather than
    # the usual "+git".
fi

orig_dir="$(pwd)"

# Tarball naming
tarball_version="$formal_version"
if $DEBIAN_NAMING; then
    tarball_version=`echo $tarball_version | sed -e 's/-/~/'`
    tarball_basename="mypaint_${tarball_version}.orig.tar"
else
    tarball_basename="mypaint-${tarball_version}.tar"
fi

# Tarball version string is used for the directory
exportdir_basename="mypaint-$tarball_version"
exportdir_location="/tmp/.mypaint-export-$$"
exportdir_path="$exportdir_location/$exportdir_basename"

# Construct release tmpdir
rm -rf "$exportdir_location"
mkdir -p "$exportdir_location"
git checkout-index -a -f --prefix="$exportdir_path/"

# Include submodules into release tarball
submodule_paths="brushlib"
git submodule update --init
for submod_path in $submodule_paths; do
    (cd "$submod_path" && \
     git checkout-index -a -f --prefix="$exportdir_path/$submod_path/")
done

# Tidy up release tmpdir, and record info in it about what was used.  If
# the release_info file exists in a build tree, scons will write it into
# the generated ./mypaint script in place of information it would
# otherwise glean from .git.
cd "$exportdir_path"
rm -f release.sh
rm -f .travis.yml
rm -fr .git*
# The format must be both valid Python and shell.
echo  >release_info "# Tarball version info, captured by release.sh"
echo >>release_info "# Base version: x.y.z, optional prerelease phase suffix"
echo >>release_info "MYPAINT_VERSION_BASE='$base_version'"
echo >>release_info "# Long version: has date suffix in prerelease phases"
echo >>release_info "MYPAINT_VERSION_FORMAL='$formal_version'"
echo >>release_info "# Extra-long version: has build/export info for prerelease"
echo >>release_info "MYPAINT_VERSION_CEREMONIAL='$long_version'"
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
    cd "$exportdir_path"
    scons debug=true
    echo "Running tests ..."
    python tests/test_mypaintlib.py
    python tests/test_compositeops.py
    python tests/test_rendering.py
    if ! $HEADLESS; then
        python tests/test_performance.py -a -c 1
        python tests/test_memory_leak.py -a -e
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

# Results, notices about tagging
$GZIP_TARBALL && ls -sSh "$tarball".gz
$BZIP2_TARBALL && ls -sSh "$tarball".bz2
ls -sSh "$tarball".xz
if $is_prerelease; then
    echo "Prereleases are generally not tagged in git, with the exception of"
    echo "some release candidates (-rcN)."
    if echo $base_version | grep -q -- "-rc"; then
        echo "Release candidate detected,"
        echo "  you can tag it with 'git tag -s v$base_version'"
    fi
else
    echo "You can tag this release with 'git tag -s v$base_version'"
fi
