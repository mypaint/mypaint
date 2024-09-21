# -*- coding: utf-8 -*-
# (c) 2003 Gustavo J A M Carneiro gjc at inescporto.pt
#     2004-2005 Filip Van Raemdonck
#
# http://www.daa.com.au/pipermail/pygtk/2003-August/005775.html
# Message-ID: <1062087716.1196.5.camel@emperor.homelinux.net>
#     "The license is whatever you want."
#
# This file was downloaded from http://www.sysfs.be/downloads/
# Adaptions 2009-2010 by Martin Renold:
# - let KeyboardInterrupt through
# - print traceback to stderr before showing the dialog
# - nonzero exit code when hitting the "quit" button
# - suppress more dialogs while one is already active
# - fix Details button when a context in the traceback is None
# - remove email features
# - fix lockup with dialog.run(), return to mainloop instead
# see also http://faq.pygtk.org/index.py?req=show&file=faq20.010.htp
# (The license is still whatever you want.)
from __future__ import division, print_function
from lib.pycompat import PY3

import inspect
import linecache
import pydoc
import sys
import traceback
from gettext import gettext as _
import textwrap

from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import Pango

import lib.meta

if PY3:
    from io import StringIO
    from urllib.parse import quote_plus
else:
    from cStringIO import StringIO
    from urllib import quote_plus


# Function that will be called when the user presses "Quit"
# Return True to confirm quit, False to cancel
quit_confirmation_func = None

RESPONSE_QUIT = 1
RESPONSE_SEARCH = 2
RESPONSE_REPORT = 3

def analyse_simple(exctyp, value, tb):
    trace = StringIO()
    traceback.print_exception(exctyp, value, tb, None, trace)
    return trace


def lookup(name, frame, lcls):
    '''Find the value for a given name in the given frame'''
    if name in lcls:
        return 'local', lcls[name]
    elif name in frame.f_globals:
        return 'global', frame.f_globals[name]
    elif '__builtins__' in frame.f_globals:
        builtins = frame.f_globals['__builtins__']
        if type(builtins) is dict:
            if name in builtins:
                return 'builtin', builtins[name]
        else:
            if hasattr(builtins, name):
                return 'builtin', getattr(builtins, name)
    return None, []


def analyse(exctyp, value, tb):
    import tokenize
    import keyword
    import platform
    from gui import application
    from gui.meta import get_libs_version_string

    app = application.get_app()

    trace = StringIO()
    nlines = 3
    frecs = inspect.getinnerframes(tb, nlines)

    trace.write('Mypaint version: %s\n' % app.version)
    trace.write('System information: %s\n' % platform.platform())
    trace.write('Using: %s\n' % (get_libs_version_string(),))

    trace.write('Traceback (most recent call last):\n')
    for frame, fname, lineno, funcname, context, cindex in frecs:
        trace.write('  File "%s", line %d, ' % (fname, lineno))
        args, varargs, varkw, lcls = inspect.getargvalues(frame)

        def readline(lno=[lineno], *args):
            if args:
                print(args)

            try:
                return linecache.getline(fname, lno[0])
            finally:
                lno[0] += 1
        all, prev, name, scope = {}, None, '', None
        for ttype, tstr, stup, etup, line in tokenize.generate_tokens(readline):
            if ttype == tokenize.NAME and tstr not in keyword.kwlist:
                if name:
                    if name[-1] == '.':
                        try:
                            val = getattr(prev, tstr)
                        except AttributeError:
                            # XXX skip the rest of this identifier only
                            break
                        name += tstr
                else:
                    assert not name and not scope
                    scope, val = lookup(tstr, frame, lcls)
                    name = tstr
                if val is not None:
                    prev = val
            elif tstr == '.':
                if prev:
                    name += '.'
            else:
                if name:
                    all[name] = (scope, prev)
                prev, name, scope = None, '', None
                if ttype == tokenize.NEWLINE:
                    break

        try:
            details = inspect.formatargvalues(args, varargs, varkw, lcls, formatvalue=lambda v: '=' + pydoc.text.repr(v))
        except:
            # seen that one on Windows (actual exception was KeyError: self)
            details = '(no details)'
        trace.write(funcname + details + '\n')
        if context is None:
            context = ['<source context missing>\n']
        trace.write(''.join(['    ' + x.replace('\t', '  ') for x in filter(lambda a: a.strip(), context)]))
        if len(all):
            trace.write('  variables: %s\n' % str(all))

    trace.write('%s: %s' % (exctyp.__name__, value))
    return trace


def _info(exctyp, value, tb):
    global exception_dialog_active
    if exctyp is KeyboardInterrupt:
        return original_excepthook(exctyp, value, tb)
    sys.stderr.write(analyse_simple(exctyp, value, tb).getvalue())
    if exception_dialog_active:
        return

    Gdk.pointer_ungrab(Gdk.CURRENT_TIME)
    Gdk.keyboard_ungrab(Gdk.CURRENT_TIME)

    exception_dialog_active = True
    # Create the dialog
    dialog = Gtk.MessageDialog(message_type=Gtk.MessageType.WARNING)
    dialog.set_title(_("Bug Detected"))

    primary = _(
        "<big><b>A programming error has been detected.</b></big>"
    )
    secondary = _(
        "You may be able to ignore this error and carry on working, "
        "but you should probably save your work soon.\n\n"
        "Please tell the developers about this using the issue tracker "
        "if no-one else has reported it yet."
    )
    dialog.set_markup(primary)
    dialog.format_secondary_text(secondary)

    dialog.add_button(_(u"Search Tracker…"), RESPONSE_SEARCH)
    if "-" in lib.meta.MYPAINT_VERSION:  # only development and prereleases
        dialog.add_button(_("Report…"), RESPONSE_REPORT)
        dialog.set_response_sensitive(RESPONSE_REPORT, False)
    dialog.add_button(_("Ignore Error"), Gtk.ResponseType.CLOSE)
    dialog.add_button(_("Quit MyPaint"), RESPONSE_QUIT)

    # Add an expander with details of the problem to the dialog
    def expander_cb(expander, *ignore):
        # Ensures that on deactivating the expander, the dialog is resized down
        if expander.get_expanded():
            dialog.set_resizable(True)
        else:
            dialog.set_resizable(False)
    details_expander = Gtk.Expander()
    details_expander.set_label(_(u"Details…"))
    details_expander.connect("notify::expanded", expander_cb)

    textview = Gtk.TextView()
    textview.show()
    textview.set_editable(False)
    textview.modify_font(Pango.FontDescription("Monospace normal"))

    sw = Gtk.ScrolledWindow()
    sw.show()
    sw.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
    sw.add(textview)

    # Set window sizing so that it's always at least 600 pixels wide, and
    # increases by 300 pixels in height once the details panel is open
    sw.set_size_request(0, 300)
    dialog.set_size_request(600, 0)

    details_expander.add(sw)
    details_expander.show_all()
    dialog.get_content_area().pack_start(details_expander, True, True, 0)

    # Get the traceback and set contents of the details
    try:
        trace = analyse(exctyp, value, tb).getvalue()
    except:
        try:
            trace = _("Exception while analyzing the exception.") + "\n"
            trace += analyse_simple(exctyp, value, tb).getvalue()
        except:
            trace = _("Exception while analyzing the exception.")
    buf = textview.get_buffer()
    trace = "\n".join(["```python", trace, "```"])
    buf.set_text(trace)
    ## Would be nice to scroll to the bottom automatically, but @#&%*@
    #first, last = buf.get_bounds()
    #buf.place_cursor(last)
    #mark = buf.get_insert()
    ##buf.scroll_mark_onscreen()
    ##textview.scroll_mark_onscreen(buf.get_insert(), 0)
    #textview.scroll_to_mark(mark, 0.0)

    # Connect callback and present the dialog
    dialog.connect('response', _dialog_response_cb, trace, exctyp, value)
    #dialog.set_modal(True) # this might actually be contra-productive...
    dialog.show()
    # calling dialog.run() here locks everything up in some cases, so
    # we just return to the main loop instead


def _dialog_response_cb(dialog, resp, trace, exctyp, value):
    global exception_dialog_active

    if resp == RESPONSE_QUIT and Gtk.main_level() > 0:
        if not quit_confirmation_func:
            sys.exit(1)  # Exit code is important for IDEs
        else:
            if quit_confirmation_func():
                sys.exit(1)  # Exit code is important for IDEs
            else:
                dialog.destroy()
                exception_dialog_active = False
    elif resp == RESPONSE_SEARCH:
        search_url = (
            "https://github.com/mypaint/mypaint/search"
            "?utf8=%E2%9C%93"
            "&q={}+{}"
            "&type=Issues"
        ).format(
            quote_plus(exctyp.__name__, "/"),
            quote_plus(str(value), "/")
        )
        Gtk.show_uri(None, search_url, Gdk.CURRENT_TIME)
        if "-" in lib.meta.MYPAINT_VERSION:
            dialog.set_response_sensitive(RESPONSE_REPORT, True)
    elif resp == RESPONSE_REPORT:
        # TRANSLATORS: Crash report template for github, preceding a traceback.
        # TRANSLATORS: Please ask users kindly to supply at least an English
        # TRANSLATORS: title if they are able.
        body = _(u"""\
            #### Description

            Give this report a short descriptive title.
            Use something like
            "[feature-that-broke]: [what-went-wrong]"
            for the title, if you can.
            Then please replace this text
            with a longer description of the bug.
            Screenshots or videos are great, too!

            #### Steps to reproduce

            Please tell us what you were doing
            when the error message popped up.
            If you can provide step-by-step instructions
            on how to reproduce the bug,
            that's even better.

            #### Traceback
        """)
        body = "\n\n".join([
            "".join(textwrap.wrap(p, sys.maxsize))
            for p in textwrap.dedent(body).split("\n\n")
        ] + [trace])
        report_url = (
            "https://github.com/mypaint/mypaint/issues/new"
            "?title={title}"
            "&body={body}"
        ).format(
            title="",
            body=quote_plus(body.encode("utf-8"), "/"),
        )
        Gtk.show_uri(None, report_url, Gdk.CURRENT_TIME)
    else:
        dialog.destroy()
        exception_dialog_active = False


original_excepthook = sys.excepthook
sys.excepthook = _info
exception_dialog_active = False


if __name__ == '__main__':
    import sys
    import os

    def _test_button_clicked_cb(*a):
        class _TestException (Exception):
            pass
        raise _TestException("That was supposed to happen.")

    win = Gtk.Window()
    win.set_size_request(200, 150)
    win.set_title(os.path.basename(sys.argv[0]))
    btn = Gtk.Button(label="Break it")
    btn.connect("clicked", _test_button_clicked_cb)
    win.add(btn)
    win.connect("destroy", lambda *a: Gtk.main_quit())
    win.show_all()
    Gtk.main()
