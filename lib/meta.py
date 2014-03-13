# This file is part of MyPaint.
# Copyright (C) 2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""GUI-independent project meta-information.

Version Strings
---------------

MyPaint now uses `Semantic Versioning`_ for its version strings.

    ``MAJOR.MINOR.PATCH[-PREREL][+BUILD]``
    
As a result, prerelease phases are marked in the code itself.
The currently defined prerelease phases are:

Suffix "-alpha":
    Actually the active development cycle.
    If any "alpha release" tarballs are made in this phase,
    they will be informal snapshots only,
    made towards the end of the cycle.

    Don't assume that the version number before this
    will actually be that of the next proper release:
    it may be bumped forward again at the start of the beta cycle
    if enough change has happened.

Suffix "-beta":
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

    A new alpha phase with an incremented ``PATCH``
    will start at the next commit.

The base version string may therefore look like:

  * "1.1.1-alpha"
  * "1.2.0-beta"
  " "1.2.0"

The release script expands the base version in the code
by appending a dot and a YYYYMMDD date string
to shows when the release was made.
It uses this for versioning the release and any tarballs made:

  * "2.4.7-beta.20190309"
  * "mypaint-2.4.7-beta.20190309.tar.bz2"

The date string is only appended
if the base version contains a prerelease suffix,
because final releases are only published once.

Whether you're building a tarball or from a working git repository,
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

"+gitexport.a23abcd"
    Build was from an exported tarball,
    which was created at the specified commit.

Again, final releases are not denoted with suffixes of any kind,
not even build information.
Porters are welcome to add their own, however.

.. _Semantic Versioning: http://semver.org/
"""

#: Base version string
#: Used by release.sh, so it must also be a line of valid POSIX shell -
#: after it has been grepped out.

MYPAINT_VERSION='1.1.1-alpha'

# 1.1.1-alpha will probably be followed by the 1.2.0-beta cycle,
# unless we bump it further for marketing reasons


