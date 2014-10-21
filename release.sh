#!/bin/bash
# Make a tarball release from git, running tests.
# This does roughly what 'make distcheck' would do if we were using autotools.
#
# Usage:
#   $ sh release.sh
#
# Environment:
#   SKIP_GITCHECK  allow running with uncommitted changes
#   SKIP_TESTS     avoid running the tests
#   SKIP_CLEANUP   don't clean up the export location
#
# Only skip phases when debugging the export process itself.

set -e

if ! test -d .git; then
    echo "Not at the root of a git repository"
    exit 1
fi
if test "x$SKIP_GITCHECK" = "x"; then
    if ! git diff --quiet; then
        echo "You have local changes, stage them first with 'git add'!"
        exit 1
    fi
fi

# Base version; a string like "1.1.0" for stable releases or "1.1.1-activedev"
# when making prereleases during the active development cycle.
base_version="`python lib/meta.py`"
echo "Version $base_version"

# Other version info we might want to capture in the tarball name or contents
rel_datestr=`date '+%Y%m%d'`
rel_gitrev=`git rev-parse --short HEAD`

# Prereleases have lengthier version strings generally, and especially long
# ones for the about box.
is_prerelease=false
tarball_version="$base_version"
long_version="$base_version"
if echo $base_version | grep -q -- "-"; then
    is_prerelease=true
    tarball_version="$base_version.$rel_datestr"
    long_version="$tarball_version+gitexport.$rel_gitrev"
    # If somebody builds from a prerelease tarball rather than from a git
    # checkout, it is marked as "+gitexport" in the about box rather than
    # the usual "+git".
fi

orig_dir="$(pwd)"
tarball_basename="mypaint-$tarball_version"
exportdir_basename="mypaint-$tarball_version"
exportdir_path="/tmp/$exportdir_basename"
tarball_output_dir="$(pwd)"

# Construct release tmpdir
# Base version name is used for the directory
rm -rf "$exportdir_path"
git checkout-index -a -f --prefix="$exportdir_path/"

# Include submodules into release tarball
submodule_paths="brushlib"
git submodule update --init
for submod_path in $submodule_paths; do
    (cd "$submod_path" && \
     git checkout-index -a -f --prefix="$exportdir_path/$submod_path/")
done

# Tidy up release tmpdir, and record info in it about what was used.  If the
# release_info file exists in a build tree, scons will use it in the generated
# ./mypaint script in place of information it would otherwise glean from .git
cd "$exportdir_path"
rm -f release.sh
rm -f .travis.yml
rm -fr .git*
echo  >release_info "# Tarball version info, captured by release.sh"
echo >>release_info "# Base version: x.y.z, optional prerelease phase suffix"
echo >>release_info "MYPAINT_VERSION_BASE = '$base_version'"
echo >>release_info "# Long version: has date suffix in prerelease phases"
echo >>release_info "MYPAINT_VERSION_FORMAL = '$tarball_version'"
echo >>release_info "# Extra-long version: has build/export info for prerelease"
echo >>release_info "MYPAINT_VERSION_CEREMONIAL = '$long_version'"
cd ..

# Create tarballs of release dir before we do any test builds
tarball="$tarball_output_dir/$exportdir_basename.tar"
rm -f "$tarball"
tar -cvf "$tarball" "$exportdir_basename"

echo "Making $tarball.gz ..."
gzip -9 --keep --force "$tarball"
echo "Making $tarball.bz2 ..."
bzip2 -9 --keep --force "$tarball"
echo "Making $tarball.xz ..."
xz -9 --force "$tarball"

# Build the release and test it
if test "x$SKIP_TESTS" = "x"; then
    echo "Making debug build inside $exportdir_path ..."
    cd "$exportdir_path"
    scons debug=true
    echo "Running tests ..."
    python tests/test_mypaintlib.py
    python tests/test_compositeops.py
    python tests/test_rendering.py
    python tests/test_performance.py -a -c 1
    python tests/test_memory_leak.py -a -e
    echo "Done testing."
else
    echo "Skipping debug build and tests (SKIP_TESTS)"
fi

# Clean up
cd "$orig_dir"
cat "$exportdir_path/release_info"
if test "x$SKIP_CLEANUP" = "x"; then
    echo "Cleaning up $exportdir_path ..."
    rm -fr "$exportdir_path"
else
    echo "Skipping cleanup of $exportdir_path (SKIP_CLEANUP)"
fi

# Results, notices about tagging
ls -sSh "$tarball".gz
ls -sSh "$tarball".bz2
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
