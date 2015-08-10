# -*- coding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2007-2014 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2015 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


import os
import time
import logging
logger = logging.getLogger(__name__)

from gi.repository import GLib
from gi.repository import Gtk


class Profiler (object):
    """Handles various kinds of profiling state for the main app.

    """

    def __init__(self):
        super(Profiler, self).__init__()
        self.profiler_active = False

    def toggle_profiling(self):
        """Starts profiling if not running, or stops it & shows results.

        """
        if self.profiler_active:
            self.profiler_active = False
        else:
            GLib.idle_add(self._do_profiling)

    def _do_profiling(self):
        """Runs the GTK main loop in the cProfile profiler till stopped."""
        import cProfile
        profile = cProfile.Profile()

        self.profiler_active = True
        logger.info('--- GUI Profiling starts ---')
        while self.profiler_active:
            profile.runcall(Gtk.main_iteration_do, False)
            if not Gtk.events_pending():
                time.sleep(0.050)  # ugly trick to remove "user does nothing" from profile
        logger.info('--- GUI Profiling ends ---')

        profile.dump_stats('profile_fromgui.pstats')
        logger.debug('profile written to mypaint_profile.pstats')
        if os.path.exists("profile_fromgui.png"):
            os.unlink("profile_fromgui.png")
        os.system('gprof2dot.py -f pstats profile_fromgui.pstats | dot -Tpng -o profile_fromgui.png')
        if os.path.exists("profile_fromgui.png"):
            os.system('xdg-open profile_fromgui.png &')
