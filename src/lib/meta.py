# This file is part of MyPaint.
# Copyright (C) 2014-2016 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""GUI-independent project meta-information.

Version Strings
---------------

MyPaint uses `Semantic Versioning`_ for its version strings.

    ``MAJOR.MINOR.PATCH[-PREREL][+BUILD]``

As a result, prerelease phases are marked in the code itself.
The currently defined prerelease phases are:

Suffix "-alpha":
    Actually the active development cycle.
    If any "alpha release" tarballs are made in this phase,
    they will be informal snapshots only,
    made towards the end of the cycle.

Suffix "-beta" and a beta release number:
    The formal beta-testing cycle.
    No new features will be added in this phase, and
    only bugfixes will be accepted.

    User interface strings are frozen too,
    so that translators can get to work. This means that
    strings in the code can now only be patched
    to fix fundamental problems in their meanings.

Suffix "-rc":
    The formal release candidate cycle.

    Only critical bugfixes and string changes are allowed.

Empty string:
    The final release commit itself, _and only that commit_.

    A new alpha phase with an incremented ``MINOR``
    will start at the next commit.

The base version string may therefore look like:

  * "1.1.1-alpha"
  * "1.2.0-beta.0"
  * "1.2.0"

The release script expands the base version in the code
by appending a dot and the number of revisions committed
since the tag of the same name (with a "v" prefix).
This tag must exist in the source.

It uses this for versioning the release and any tarballs made:

  * "mypaint-2.4.7-beta.0.4.tar.bz2"
    - ad-hoc beta release four revisions after the "v2.4.7-beta.0" tag.
  * "2.4.7"
    - release 2.4.7 itself!

Whether you're building from a release tarball
or from a working git repository,
the build scripts also collect information about
when and where the build is taking place,
and from what release or git version.
They further append this build-level information to the version string,
in a way that makes it available at runtime.
This provides SemVer-style "+BUILD" notation
for display in the about box.
Some build suffixes and their meanings:

"+git.123abcd"
    Build was direct from a git repository,
    and was made at the specified commit.

"+gitexport.123abcd"
    Build was from an exported tarball,
    which was created at the specified commit.

"+git.123abcd.dirty"
    Build was direct from a git repository,
    which was created at the specified commit,
    but there were local uncommitted changes.

This build information is always present in the long about box version
number, but is never present in tarball names or other released
artifacts.

.. _Semantic Versioning: http://semver.org/

"""
from __future__ import division, print_function

#: Program name, for display.
#: Not marked for translation, but that can change if it enhances things.

MYPAINT_PROGRAM_NAME = "MyPaint"

ALPHA = '-alpha'
BETA = '-beta'
RC = '-rc'
VALID_PRERELEASE_VALUES = {'', ALPHA, BETA, RC}

MAJOR = 2
MINOR = 1
PATCH = 0
PREREL = ALPHA
PREREL_NUM = 0

# Verify the version fields
for part in (MAJOR, MINOR, PATCH, PREREL_NUM):
    assert isinstance(part, int) and part >= 0
assert PREREL in VALID_PRERELEASE_VALUES
if PREREL == ALPHA:
    assert PREREL_NUM == 0

#: Base version string.
#: This is required to match a tag in git for formal releases. However
#: for pre-release (hyphenated) base versions, the formal version will
#: be further decorated with the number of commits following the tag.
MYPAINT_VERSION = '{major}.{minor}.{patch}{prerel}'.format(
    major=MAJOR, minor=MINOR, patch=PATCH,
    prerel=PREREL and
    # Prerelease numbers should only be used for beta releases
    PREREL + ('.' + str(PREREL_NUM) if PREREL in {BETA, RC} else '')
)


def _parse_version_string(version_string):
    """Parse version string into fields

    If the string is not a valid version string, None is returned.

    :param version_string: version string to parse
    :return: the four version fields as a tuple, or None if input is invalid
    :rtype: (int, int, int, str) | None

    >>> # Invalid input strings
    >>> _parse_version_string("1.0.2.3-alpha")
    >>> _parse_version_string('2.0-beta')
    >>> _parse_version_string('2.0.3-gamma')
    >>> _parse_version_string('1.-2.3')
    >>> # Valid input strings
    >>> _parse_version_string("1.0.2-alpha")
    (1, 0, 2, '-alpha')
    >>> _parse_version_string('3.1.5')
    (3, 1, 5, '')
    >>> _parse_version_string('2.0.1-alpha')
    (2, 0, 1, '-alpha')
    >>> _parse_version_string('2.2.1-rc')
    (2, 2, 1, '-rc')
    """
    if '-' in version_string:
        i = version_string.index('-')
        prerel = version_string[i:]
        version_string = version_string[:i]
        # Strip prerelease number
        if '.' in prerel:
            prerel = prerel[:prerel.index('.')]
    else:
        prerel = ''
    try:
        assert prerel in VALID_PRERELEASE_VALUES
        major, minor, patch = (int(f) for f in version_string.split('.'))
        return major, minor, patch, prerel
    except (ValueError, AssertionError):
        return None


class Compatibility:
    """ Enum-like class holding only compatibility type constants
    """
    # app major version < file major version
    INCOMPATIBLE = 1
    # app major version = file major version and
    # app minor version < file minor version
    PARTIALLY = 2
    # app major version >= file major version and
    # app minor version >= file minor version
    FULLY = 3

    DESC = {
        INCOMPATIBLE: 'incompatible',
        PARTIALLY: 'only partially compatible',
        FULLY: 'compatible',
    }


def compatibility(target_version_string):
    """ Check if the current version is compatible

    :param target_version_string: Version string to test against
    :return: The compatibility of the current version with the target version
       as a tuple of a compatibility type constant and a boolean indicating
       whether the target version is a prerelease.
    :rtype: (int, bool) | None
    """
    target = _parse_version_string(target_version_string)
    return target and _compatibility(target, (MAJOR, MINOR, PATCH, PREREL))


def _compatibility(target_version_fields, current_version_fields):
    """ Internal implementation of version compatibility check

    >>> C = Compatibility
    >>> _compatibility((1,0,0,''), (1,0,0,'')) == (C.FULLY, False)
    True
    >>> _compatibility((1,0,0,'-alpha'), (1,0,0,'')) == (C.FULLY, True)
    True
    >>> _compatibility((1,0,0,''), (1,0,0,'-beta')) == (C.FULLY, False)
    True
    >>> _compatibility((2,0,1,''), (2,1,0,'-beta')) == (C.FULLY, False)
    True
    >>> _compatibility((2,3,1,''), (2,2,5,'')) == (C.PARTIALLY, False)
    True
    >>> _compatibility((1,3,1,'-alpha'), (1,3,0,'')) == (C.FULLY, True)
    True
    >>> _compatibility((2,0,1,'-alpha'), (1,3,0,'')) == (C.INCOMPATIBLE, True)
    True
    """
    t_major, t_minor, t_patch, t_prerel = target_version_fields
    c_major, c_minor, c_patch, c_prerel = current_version_fields
    C = Compatibility
    if t_major > c_major:
        comp = C.INCOMPATIBLE
    elif t_major < c_major:
        comp = C.FULLY
    elif t_minor < c_minor:
        comp = C.FULLY
    elif t_minor > c_minor:
        comp = C.PARTIALLY
    else:
        comp = C.FULLY
    return (comp, t_prerel != '')


# Release building magic

def _get_versions(gitprefix="gitexport"):
    """Gets all version strings for use in release/build scripting.

    :param str gitprefix: how to denote git-derived build metainfo
    :rtype: tuple
    :returns: all 3 version strings: (base, formal, ceremonial)

    This function must only be called by Python build scripts,
    or by release.sh (which invokes the interpreter).

    It assumes that the current working directory is either the
    one-level-down directory in an unpacked export generated by
    release.sh (when a `release_info` file exists), or a working git
    repository (when a `.git` directory exists).

    The `gitprefix` is only used when examining the local git repository
    to derive the additional information.

    """
    import re
    import os
    import sys
    import subprocess
    # Establish some fallbacks for use when there's no .git present,
    # or no release_info.
    base_version = MYPAINT_VERSION
    formal_version = base_version
    ceremonial_version = formal_version + "+unknown"
    if os.path.isfile("release_info"):
        # If release information from release.sh exists, use that
        relinfo = {}
        with open("release_info", "rb") as relinfo_fp:
            exec(relinfo_fp.read(), relinfo)
        base_version = relinfo.get(
            "MYPAINT_VERSION_BASE",
            base_version,
        )
        formal_version = relinfo.get(
            "MYPAINT_VERSION_FORMAL",
            formal_version,
        )
        ceremonial_version = relinfo.get(
            "MYPAINT_VERSION_CEREMONIAL",
            ceremonial_version,
        )
    elif base_version.endswith(ALPHA):
        # There will be no matching git tag for initial alpha (active
        # development) phases.
        if os.path.isdir(".git"):
            cmd = ["git", "rev-parse", "--short", "HEAD"]
            try:
                objsha = subprocess.check_output(cmd, universal_newlines=True)
            except (OSError, subprocess.CalledProcessError):
                print("ERROR: Failed to invoke %r. Build will be marked as "
                      "unsupported." % (" ".join(cmd), ),
                      file=sys.stderr)
            else:
                build_ids = [gitprefix, objsha.strip()]
                build_metadata = ".".join(build_ids)
                ceremonial_version = "{}+{}".format(
                    formal_version,
                    build_metadata,
                )
    elif os.path.isdir(".git"):
        # Pull the additional info from git.
        cmd = ["git", "describe", "--tags", "--long", "--dirty", "--always"]
        try:
            git_desc = subprocess.check_output(cmd, universal_newlines=True)
        except (OSError, subprocess.CalledProcessError):
            print("ERROR: Failed to invoke %r. Build will be marked as "
                  "unsupported." % (" ".join(cmd), ),
                  file=sys.stderr)
        else:
            git_desc = git_desc.strip()
            # If MYPAINT_VERSION matches the most recent tag in git,
            # then use the extra information from `git describe`.
            parse_pattern = r'''
                ^ v{base_version}   #  Expected base version.
                (?:-(\d+))?         #1 Number of commits since the tag.
                (?:-g([0-9a-f]+))?  #2 Abbr'd SHA of the git tree exported.
                (?:-(dirty))?       #3 Highlight uncommitted changes.
                $
            '''.rstrip().format(base_version=re.escape(base_version))
            parse_re = re.compile(parse_pattern, re.VERBOSE | re.IGNORECASE)
            match = parse_re.match(git_desc)
            objsha = None
            nrevs = 0
            dirty = None
            if match:
                (nrevs, objsha, dirty) = match.groups()
            else:
                print("WARNING: Failed to parse output of \"{cmd}\". "
                      "The base MYPAINT_VERSION ({ver}) from the code "
                      "should be present in the output of this command "
                      "({git_desc}).".format(cmd=" ".join(cmd),
                                             git_desc=repr(git_desc),
                                             ver=base_version),
                      file=sys.stderr)
                print("HINT: make sure you have the most recent tags: "
                      "git fetch --tags",
                      file=sys.stderr)
                cmd = ["git", "rev-parse", "--short", "HEAD"]
                print(
                    "WARNING: falling back to using just \"{cmd}\".".format(
                        cmd=" ".join(cmd)),
                    file=sys.stderr)
                try:
                    cmdout = subprocess.check_output(
                        cmd, universal_newlines=True)
                except (OSError, subprocess.CalledProcessError):
                    print("ERROR: Failed to invoke %r. Build will be marked "
                          "as unsupported." % (" ".join(cmd), ),
                          file=sys.stderr)
                else:
                    cmdout = cmdout.strip()
                if re.match(r'^([0-9a-f]{7,})$', cmdout, re.I):
                    objsha = cmdout
                else:
                    print("WARNING: Output of {cmd} ({output}) does not look "
                          "like a git revision SHA.".format(cmd=" ".join(cmd),
                                                            output=cmdout),
                          file=sys.stderr)
            # nrevs is None or zero if this commit is the matched tag.
            # If not, then incorporate the numbers somehow.
            if nrevs and int(nrevs) > 0:
                if "-" not in base_version:
                    raise ValueError(
                        "The code's MYPAINT_VERSION ({ver}) "
                        "denotes a final release but there are commits "
                        "after the tag v{ver} in this git branch. "
                        "A new 'vX.Y.Z-alpha' phase tag needs to be "
                        "created for the next version now, "
                        "and lib.meta.MYPAINT_VERSION needs to be "
                        "updated to match it."
                        .format(
                            ver=base_version,
                        )
                    )
                    # Can't just fake it with a hyphen: that would
                    # have lower priority than the final release.
                else:
                    # It's already something like "1.2.0-alpha",
                    # so we can use a dot-suffix: "1.2.0-alpha.42".
                    formal_version = "%s.%s" % (base_version, nrevs)
                # The super-long version may be overridden later too,
                # but for now it must incorporate the normal long
                # version string.
                ceremonial_version = formal_version
            # Collect details about the build after a plus sign.
            # objsha is None if this commit is the matched tag.
            # The dirty flag is only present if there are uncommitted
            # changes (which shouldn't happen).
            build_ids = []
            if objsha:
                build_ids.append(gitprefix + "." + objsha)
            if dirty:
                build_ids.append("dirty")
            if build_ids:
                build_metadata = ".".join(build_ids)
                ceremonial_version = "{}+{}".format(
                    formal_version,
                    build_metadata,
                )
    else:
        pass
    return (base_version, formal_version, ceremonial_version)


def _get_release_info_script(gitprefix="gitexport"):
    """Returns a script fragment describing the release.

    Like _get_versions(), this must only be called from build scripting
    or similar machinery. The returned string can be processed by either
    Python or Bourne Shell.

    """
    base, formal, ceremonial = _get_versions(gitprefix=gitprefix)
    release_info = "MYPAINT_VERSION_BASE=%r\n" % (base,)
    release_info += "MYPAINT_VERSION_FORMAL=%r\n" % (formal,)
    release_info += "MYPAINT_VERSION_CEREMONIAL=%r\n" % (ceremonial,)
    return release_info


# release.sh expects to be able to run this file as __main__, and uses
# it to generate the release_info script in the release tarball it
# makes.

if __name__ == '__main__':
    print(_get_release_info_script(), end=' ')
