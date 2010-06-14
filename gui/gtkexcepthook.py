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

import inspect, linecache, pydoc, sys, traceback
from cStringIO import StringIO
from gettext import gettext as _

import pygtk
pygtk.require ('2.0')
import gtk, pango

def analyse_simple (exctyp, value, tb):
    trace = StringIO()
    traceback.print_exception (exctyp, value, tb, None, trace)
    return trace

def lookup (name, frame, lcls):
    '''Find the value for a given name in the given frame'''
    if name in lcls:
        return 'local', lcls[name]
    elif name in frame.f_globals:
        return 'global', frame.f_globals[name]
    elif '__builtins__' in frame.f_globals:
        builtins = frame.f_globals['__builtins__']
        if type (builtins) is dict:
            if name in builtins:
                return 'builtin', builtins[name]
        else:
            if hasattr (builtins, name):
                return 'builtin', getattr (builtins, name)
    return None, []

def analyse (exctyp, value, tb):
    import tokenize, keyword

    trace = StringIO()
    nlines = 3
    frecs = inspect.getinnerframes (tb, nlines)
    trace.write ('Traceback (most recent call last):\n')
    for frame, fname, lineno, funcname, context, cindex in frecs:
        trace.write ('  File "%s", line %d, ' % (fname, lineno))
        args, varargs, varkw, lcls = inspect.getargvalues (frame)

        def readline (lno=[lineno], *args):
            if args: print args
            try: return linecache.getline (fname, lno[0])
            finally: lno[0] += 1
        all, prev, name, scope = {}, None, '', None
        for ttype, tstr, stup, etup, line in tokenize.generate_tokens (readline):
            if ttype == tokenize.NAME and tstr not in keyword.kwlist:
                if name:
                    if name[-1] == '.':
                        try:
                            val = getattr (prev, tstr)
                        except AttributeError:
                            # XXX skip the rest of this identifier only
                            break
                        name += tstr
                else:
                    assert not name and not scope
                    scope, val = lookup (tstr, frame, lcls)
                    name = tstr
                if val is not None:
                    prev = val
                #print '  found', scope, 'name', name, 'val', val, 'in', prev, 'for token', tstr
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
            details = inspect.formatargvalues (args, varargs, varkw, lcls, formatvalue=lambda v: '=' + pydoc.text.repr (v))
        except:
            # seen that one on Windows (actual exception was KeyError: self)
            details = '(no details)'
        trace.write (funcname + details + '\n')
        if context is None:
            context = ['<source context missing>\n']
        trace.write (''.join (['    ' + x.replace ('\t', '  ') for x in filter (lambda a: a.strip(), context)]))
        if len (all):
            trace.write ('  variables: %s\n' % str (all))

    trace.write ('%s: %s' % (exctyp.__name__, value))
    return trace

def _info (exctyp, value, tb):
    global exception_dialog_active
    if exctyp is KeyboardInterrupt:
        return original_excepthook(exctyp, value, tb)
    sys.stderr.write(analyse_simple (exctyp, value, tb).getvalue())
    if exception_dialog_active:
        return

    gtk.gdk.pointer_ungrab()
    gtk.gdk.keyboard_ungrab()

    exception_dialog_active = True
    dialog = gtk.MessageDialog (parent=None, flags=0, type=gtk.MESSAGE_WARNING, buttons=gtk.BUTTONS_NONE)
    dialog.set_title (_("Bug Detected"))
    if gtk.check_version (2, 4, 0) is not None:
        dialog.set_has_separator (False)

    primary = _("<big><b>A programming error has been detected.</b></big>")
    secondary = _("It probably isn't fatal, but the details should be reported to the developers nonetheless.")

    try:
        setsec = dialog.format_secondary_text
    except AttributeError:
        raise
        dialog.vbox.get_children()[0].get_children()[1].set_markup ('%s\n\n%s' % (primary, secondary))
        #lbl.set_property ("use-markup", True)
    else:
        del setsec
        dialog.set_markup (primary)
        dialog.format_secondary_text (secondary)

    dialog.add_button (_("Details..."), 2)
    dialog.add_button (gtk.STOCK_CLOSE, gtk.RESPONSE_CLOSE)
    dialog.add_button (gtk.STOCK_QUIT, 1)

    try:
        trace = analyse (exctyp, value, tb).getvalue()
    except:
        try:
            trace = _("Exception while analyzing the exception.") + "\n"
            trace += analyse_simple (exctyp, value, tb).getvalue()
        except:
            trace = _("Exception while analyzing the exception.")
        
    dialog.connect('response', _dialog_response_cb, trace)
    #dialog.set_modal(True) # this might actually be contra-productive...
    dialog.show()
    # calling dialog.run() here locks everything up in some cases, so
    # we just return to the main loop instead

def _dialog_response_cb(dialog, resp, trace):
    if resp == 2:
        # Show details...
        details = gtk.Dialog (_("Bug Details"), dialog,
          gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT,
          (gtk.STOCK_CLOSE, gtk.RESPONSE_CLOSE, ))
        details.set_property ("has-separator", False)

        textview = gtk.TextView(); textview.show()
        textview.set_editable (False)
        textview.modify_font (pango.FontDescription ("Monospace"))

        sw = gtk.ScrolledWindow(); sw.show()
        sw.set_policy (gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        sw.add (textview)
        details.vbox.add (sw)
        textbuffer = textview.get_buffer()
        textbuffer.set_text (trace)

        monitor = gtk.gdk.screen_get_default ().get_monitor_at_window (dialog.window)
        area = gtk.gdk.screen_get_default ().get_monitor_geometry (monitor)
        try:
            w = area.width // 1.6
            h = area.height // 1.6
        except SyntaxError:
            # python < 2.2
            w = area.width / 1.6
            h = area.height / 1.6
        details.set_default_size (int (w), int (h))

        details.run()
        details.destroy()

    elif resp == 1 and gtk.main_level() > 0:
        sys.exit(1) # Exit code is important for IDEs

    else:
        dialog.destroy()
        global exception_dialog_active
        exception_dialog_active = False


original_excepthook = sys.excepthook
sys.excepthook = _info
exception_dialog_active = False

