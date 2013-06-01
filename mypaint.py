# This file is part of MyPaint.
# Copyright (C) 2007-2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""
This script does all the platform dependent stuff. Its main task is
to figure out where the python modules are.
"""
import sys, os
import re
import logging
logger = logging.getLogger('mypaint')


class ColorFormatter (logging.Formatter):
    """Minimal ANSI formatter, for use with non-Windows console logging."""

    # ANSI control sequences for various things
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
    FG = 30
    BG = 40
    LEVELCOL = {
            "DEBUG": "\033[%02dm" % (FG+BLUE,),
            "INFO": "\033[%02dm" % (FG+GREEN,),
            "WARNING": "\033[%02dm" % (FG+YELLOW,),
            "ERROR": "\033[%02dm" % (FG+RED,),
            "CRITICAL": "\033[%02d;%02dm" % (FG+RED, BG+BLACK),
        }
    BOLD = "\033[01m"
    BOLDOFF = "\033[22m"
    ITALIC = "\033[03m"
    ITALICOFF = "\033[23m"
    UNDERLINE = "\033[04m"
    UNDERLINEOFF = "\033[24m"
    RESET = "\033[0m"

    # Replace tokens in message format strings to highlight interpolations
    REPLACE_BOLD = lambda m: ( ColorFormatter.BOLD +
                               m.group(0) +
                               ColorFormatter.BOLDOFF )
    REPLACE_UNDERLINE = lambda m: ( ColorFormatter.UNDERLINE +
                                    m.group(0) +
                                    ColorFormatter.UNDERLINEOFF )
    TOKEN_FORMATTING = [
            (re.compile(r'%r'), REPLACE_BOLD),
            (re.compile(r'%s'), REPLACE_BOLD),
            (re.compile(r'%\+?[0-9.]*d'), REPLACE_BOLD),
            (re.compile(r'%\+?[0-9.]*f'), REPLACE_BOLD),
        ]


    def format(self, record):
        record = logging.makeLogRecord(record.__dict__)
        msg = record.msg
        for token_re, repl in self.TOKEN_FORMATTING:
            msg = token_re.sub(repl, msg)
        record.msg = msg
        record.reset = self.RESET
        record.bold = self.BOLD
        record.boldOff = self.BOLDOFF
        record.italic = self.ITALIC
        record.italicOff = self.ITALICOFF
        record.underline = self.UNDERLINE
        record.underlineOff = self.UNDERLINEOFF
        record.levelCol = ""
        if record.levelname in self.LEVELCOL:
            record.levelCol = self.LEVELCOL[record.levelname]
        return super(ColorFormatter, self).format(record)


def win32_unicode_argv():
    # fix for https://gna.org/bugs/?17739
    # code mostly comes from http://code.activestate.com/recipes/572200/
    """Uses shell32.GetCommandLineArgvW to get sys.argv as a list of Unicode
    strings.

    Versions 2.x of Python don't support Unicode in sys.argv on
    Windows, with the underlying Windows API instead replacing multi-byte
    characters with '?'.
    """
    try:
        from ctypes import POINTER, byref, cdll, c_int, windll
        from ctypes.wintypes import LPCWSTR, LPWSTR

        GetCommandLineW = cdll.kernel32.GetCommandLineW
        GetCommandLineW.argtypes = []
        GetCommandLineW.restype = LPCWSTR
        CommandLineToArgvW = windll.shell32.CommandLineToArgvW
        CommandLineToArgvW.argtypes = [LPCWSTR, POINTER(c_int)]

        CommandLineToArgvW.restype = POINTER(LPWSTR)
        cmd = GetCommandLineW()
        argc = c_int(0)
        argv = CommandLineToArgvW(cmd, byref(argc))
        if argc.value > 0:
            # Remove Python executable if present
            if argc.value - len(sys.argv) == 1:
                start = 1
            else:
                start = 0
            return [argv[i] for i in xrange(start, argc.value)]
    except Exception:
        return [s.decode(sys.getfilesystemencoding()) for s in args]


def get_paths():
    join = os.path.join

    # Convert sys.argv to a list of unicode objects
    # (actually converting sys.argv confuses gtk, thus we add a new variable)
    if sys.platform == 'win32':
        sys.argv_unicode = win32_unicode_argv()
    else:
        sys.argv_unicode = [s.decode(sys.getfilesystemencoding())
                            for s in sys.argv]

    # Script and its location, in canonical absolute form
    scriptfile = os.path.abspath(os.path.normpath(sys.argv_unicode[0]))
    scriptdir = os.path.dirname(scriptfile)
    assert isinstance(scriptfile, unicode)
    assert isinstance(scriptdir, unicode)

    # Determine $prefix
    dir_install = scriptdir
    if os.path.basename(dir_install) == 'bin':
        # This is a normal POSIX installation.
        prefix = os.path.dirname(dir_install)
        assert isinstance(prefix, unicode)
        libpath = join(prefix, 'share', 'mypaint')
        libpath_compiled = join(prefix, 'lib', 'mypaint') # or lib64?
        sys.path.insert(0, libpath)
        sys.path.insert(0, libpath_compiled)
        localepath = join(prefix, 'share', 'locale')
        localepath_brushlib = localepath
        extradata = join(prefix, 'share')
    elif sys.platform == 'win32':
        prefix=None
        # this is py2exe point of view, all executables in root of installdir
        libpath = os.path.realpath(scriptdir)
        sys.path.insert(0, libpath)
        localepath = join(libpath, 'share', 'locale')
        localepath_brushlib = localepath
        extradata = join(libpath, 'share')
    else:
        # Not installed: run out of the source tree.
        prefix = None
        libpath = u'.'
        extradata = u'desktop'
        localepath = 'po'
        localepath_brushlib = 'brushlib/po'

    assert isinstance(libpath, unicode)

    try: # just for a nice error message
        from lib import mypaintlib
    except ImportError:
        logger.critical("We are not correctly installed or compiled!")
        logger.critical('script: %r', sys.argv[0])
        if prefix:
            logger.critical('deduced prefix: %r', prefix)
            logger.critical('lib_shared: %r', libpath)
            logger.critical('lib_compiled: %r', libpath_compiled)
        raise

    # Ensure that pyGTK compatibility is setup before anything else
    from gui import gtk2compat

    datapath = libpath
    if not os.path.isdir(join(datapath, 'brushes')):
        logger.critical('Default brush collection not found!')
        logger.critical('It should have been here: %r', datapath)
        sys.exit(1)

    # Old style config file and user data locations.
    # Return None if using XDG will be correct.
    if sys.platform == 'win32':
        old_confpath = None
    else:
        from lib import helpers
        homepath =  helpers.expanduser_unicode(u'~')
        old_confpath = join(homepath, '.mypaint/')

    if old_confpath:
        if not os.path.isdir(old_confpath):
            old_confpath = None
        else:
            logger.info("Found old-style configuration in %r", old_confpath)
            logger.info("This can be migrated to $XDG_CONFIG_HOME and "
                        "$XDG_DATA_HOME if you wish.")
            logger.info("See the XDG Base Directory Specification for info.")

    assert isinstance(old_confpath, unicode) or old_confpath is None
    assert isinstance(datapath, unicode)
    assert isinstance(extradata, unicode)

    return datapath, extradata, old_confpath, localepath, localepath_brushlib


def psyco_opt():
    # This helps on slow PCs where the python overhead dominates.
    # (30% higher framerate measured on 533MHz CPU; startup slowdown below 20%)
    # Note: python -O -O does not help.

    try:
        import psyco
        if sys.platform == 'win32':
            if psyco.hexversion >= 0x020000f0 :
                psyco.full()
                logger.info('Psyco being used')
            else:
                logger.warning("Need at least psyco 2.0 to run")
        else:
            psyco.full()
            logger.info('Psyco being used')
    except ImportError:
        pass


if __name__ == '__main__':
    # Console logging
    log_format = "%(levelname)s: %(name)s: %(message)s"
    if sys.platform == 'win32':
        # Windows doesn't understand ANSI by default.
        console_handler = logging.StreamHandler(stream=sys.stderr)
        console_formatter = logging.formatter(log_format)
    else:
        # Assume POSIX.
        # Clone stderr so that later reassignment of sys.stderr won't affect
        # logger if --logfile is used.
        stderr_fd = os.dup(sys.stderr.fileno())
        stderr_fp = os.fdopen(stderr_fd, 'ab', 0)
        # Pretty colours.
        console_handler = logging.StreamHandler(stream=stderr_fp)
        if stderr_fp.isatty():
            log_format = (
                "%(levelCol)s%(levelname)s: "
                "%(bold)s%(name)s%(reset)s%(levelCol)s: "
                "%(message)s%(reset)s" )
            console_formatter = ColorFormatter(log_format)
        else:
            console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logging_level = logging.INFO
    if os.environ.get("MYPAINT_DEBUG", False):
        logging_level = logging.DEBUG
    root_logger = logging.getLogger(None)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging_level)
    if logging_level == logging.DEBUG:
        logger.info("Debugging output enabled via MYPAINT_DEBUG")

    # Psyco setup
    psyco_opt()

    # Path determination
    datapath, extradata, old_confpath, localepath, localepath_brushlib \
        = get_paths()

    # Locale setting
    # must be done before importing any translated python modules
    # (to get global strings translated, especially brushsettings.py)
    import gettext
    import locale
    if sys.platform == 'win32':
        os.environ['LANG'] = locale.getdefaultlocale()[0]

    # Internationalization voodoo
    # https://bugzilla.gnome.org/show_bug.cgi?id=574520#c26
    #locale.setlocale(locale.LC_ALL, '')  #needed?
    logger.debug("getlocale(): %r", locale.getlocale())
    logger.debug("localepath: %r", localepath)
    logger.debug("localepath_brushlib: %r", localepath_brushlib)

    # Low-level bindtextdomain, required for GtkBuilder stuff.
    locale.bindtextdomain("mypaint", localepath)
    locale.bindtextdomain("libmypaint", localepath_brushlib)
    locale.textdomain("mypaint")

    # Python gettext module.
    # See http://docs.python.org/release/2.7/library/locale.html
    gettext.bindtextdomain("mypaint", localepath)
    gettext.bindtextdomain("libmypaint", localepath_brushlib)
    gettext.textdomain("mypaint")

    from gui import main
    version = main.MYPAINT_VERSION
    if version.endswith("+git"):
        try:
            version += _MYPAINT_BUILD_GIT_REVISION
        except NameError:
            pass

    # Start the app.
    main.main(datapath, extradata, old_confpath, version)
