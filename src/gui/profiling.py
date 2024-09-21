# -*- coding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2007-2014 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2015 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from __future__ import division, print_function

import os
import sys
import time
import tempfile
import subprocess
import shutil
import logging

from lib.gibindings import GLib
from lib.gibindings import Gtk

import lib.fileutils


logger = logging.getLogger(__name__)


# Copied from Python 3.5's distutils/spawn.py

def find_executable(executable, path=None):
    """Tries to find 'executable' in the directories listed in 'path'.

    A string listing directories separated by 'os.pathsep'; defaults to
    os.environ['PATH'].  Returns the complete filename or None if not found.
    """
    if path is None:
        path = os.environ['PATH']

    paths = path.split(os.pathsep)
    base, ext = os.path.splitext(executable)

    if (sys.platform == 'win32') and (ext != '.exe'):
        executable = executable + '.exe'

    if not os.path.isfile(executable):
        for p in paths:
            f = os.path.join(p, executable)
            if os.path.isfile(f):
                # the file exists, we have a shot at spawn working
                return f
        return None
    else:
        return executable


class Profiler (object):
    """Handles profiling state for the main app.

    The profiler's output is written to a tempdir, which is shown to the
    user in their file browser once processing is complete. Ideally
    you'll have gprof2dot.py and graphviz installed, so you can compare
    pretty PNG files directly. However even in the absence of these
    tools, you can copy the output .pstats files elsewhere and run
    gprof2dot.py and dot on them manually.

    The tempdir is deleted when the application exits normally.

    """

    GPROF2DOT = ["gprof2dot.py", "-f", "pstats"]
    DOT2PNG = ["dot", "-Tpng"]

    if find_executable("gprof2dot"):
        GPROF2DOT = ["gprof2dot", "-f", "pstats"]

    def __init__(self):
        super(Profiler, self).__init__()
        self.profiler_active = False
        self.profile_num = 0
        self.__temp_dir = None

    def toggle_profiling(self):
        """Starts profiling if not running, or stops it & shows results."""
        if self.profiler_active:
            self.profiler_active = False
        else:
            GLib.idle_add(self._do_profiling)

    @property
    def _tempdir(self):
        td = self.__temp_dir
        if not td:
            td = tempfile.mkdtemp(prefix="mypaint-profile-")
            self.__temp_dir = td
        return td

    def _do_profiling(self):
        """Runs the GTK main loop in the cProfile profiler till stopped."""
        self.profile_num += 1
        basename = "{isotime}-{n}".format(
            isotime = time.strftime("%Y%m%d-%H%M%S"),
            n = self.profile_num,
        )

        import cProfile
        profile = cProfile.Profile()

        self.profiler_active = True
        logger.info('--- GUI Profiling starts ---')
        while self.profiler_active:
            profile.runcall(Gtk.main_iteration_do, False)
            if not Gtk.events_pending():
                time.sleep(0.050)
                # ugly trick to remove "user does nothing" from profile
        logger.info('--- GUI Profiling ends ---')

        pstats_filepath = os.path.join(self._tempdir, basename + ".pstats")
        if os.path.exists(pstats_filepath):
            os.unlink(pstats_filepath)
        profile.dump_stats(pstats_filepath)
        logger.debug('profile written to %r', pstats_filepath)

        try:
            dot_filepath = os.path.join(self._tempdir, basename + ".dot")
            if os.path.exists(dot_filepath):
                os.unlink(dot_filepath)
            cmd = list(self.GPROF2DOT) + ["-o", dot_filepath, pstats_filepath]
            logger.debug("Running %r...", cmd)
            subprocess.check_call(cmd)

            png_filepath = os.path.join(self._tempdir, basename + ".png")
            if os.path.exists(png_filepath):
                os.unlink(png_filepath)
            cmd = list(self.DOT2PNG) + ["-o", png_filepath, dot_filepath]
            logger.debug("Running %r...", cmd)
            subprocess.check_call(cmd)
        except Exception:
            logger.exception(
                "Profiling output post-processing failed."
            )
            logger.info(
                "This is normal if %r and/or graphviz's %r "
                "are not both in your PATH and executable.",
                self.GPROF2DOT[0],
                self.DOT2PNG[0],
            )
        else:
            logger.info("Profiling post-processing succeeded.")
        logger.info(
            "Opening output folder %r with the default directory viewer...",
            self._tempdir,
        )
        lib.fileutils.startfile(self._tempdir)

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        """Cleans up the tempdir associated with the profile object."""
        if self.__temp_dir is None:
            return
        try:
            if os.path.isdir(self.__temp_dir):
                logger.info("Cleaning up %r...", self.__temp_dir)
                shutil.rmtree(self.__temp_dir, ignore_errors=True)
        except Exception:
            logger.exception("Cleanup of %r failed", self.__temp_dir)
        else:
            self.__temp_dir = None
