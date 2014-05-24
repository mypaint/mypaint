# -*- coding: utf-8 -*-
#
# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Main drawing window.

Painting is done in tileddrawwidget.py.
"""

## Imports

import os
import time
import webbrowser
from warnings import warn
import logging
logger = logging.getLogger(__name__)
import math

from gettext import gettext as _
import gtk
import gobject
from gtk import gdk
from gtk import keysyms

import colorselectionwindow
import historypopup
import stategroup
import colorpicker
import windowing
import toolbar
import dialogs
import layermodes
from lib import helpers
import canvasevent
from colors import RGBColor, HSVColor

import brushselectionwindow

import gtk2compat
import xml.etree.ElementTree as ET

# palette support
from lib.scratchpad_palette import GimpPalette, draw_palette

from overlays import LastPaintPosOverlay, ScaleOverlay
from framewindow import FrameOverlay
from symmetry import SymmetryOverlay


## Module constants

BRUSHPACK_URI = 'http://wiki.mypaint.info/index.php?title=Brush_Packages/redirect_mypaint_1.1_gui'


## Helpers

def with_wait_cursor(func):
    """python decorator that adds a wait cursor around a function"""
    # TODO: put in a helper file?
    def wrapper(self, *args, **kwargs):
        toplevels = gtk.Window.list_toplevels()
        toplevels = [t for t in toplevels if t.get_window() is not None]
        for toplevel in toplevels:
            toplevel_win = toplevel.get_window()
            if toplevel_win is not None:
                toplevel_win.set_cursor(gdk.Cursor(gdk.WATCH))
            toplevel.set_sensitive(False)
        self.app.doc.tdw.grab_add()
        try:
            func(self, *args, **kwargs)
            # gtk main loop may be called in here...
        finally:
            for toplevel in toplevels:
                toplevel.set_sensitive(True)
                # ... which is why we need this check:
                toplevel_win = toplevel.get_window()
                if toplevel_win is not None:
                    toplevel_win.set_cursor(None)
            self.app.doc.tdw.grab_remove()
    return wrapper


## Class definitions


class DrawWindow (gtk.Window):
    """Main drawing window.
    """

    __gtype_name__ = 'MyPaintDrawWindow'

    #TRANSLATORS: footer icon tooltip markup for the current mode
    _MODE_ICON_TEMPLATE = _("<b>{name}</b>\n{description}")

    def __init__(self):
        super(DrawWindow, self).__init__()

        import application
        app = application.get_app()
        self.app = app
        self.app.kbm.add_window(self)

        # Window handling
        self._updating_toggled_item = False
        self.is_fullscreen = False

        # Enable drag & drop
        if not gtk2compat.USE_GTK3:
            self.drag_dest_set(gtk.DEST_DEFAULT_MOTION |
                            gtk.DEST_DEFAULT_HIGHLIGHT |
                            gtk.DEST_DEFAULT_DROP,
                            [("text/uri-list", 0, 1),
                             ("application/x-color", 0, 2)],
                            gtk.gdk.ACTION_DEFAULT|gtk.gdk.ACTION_COPY)

        # Connect events
        self.connect('delete-event', self.quit_cb)
        self.connect('key-press-event', self.key_press_event_cb)
        self.connect('key-release-event', self.key_release_event_cb)
        self.connect("drag-data-received", self.drag_data_received)
        self.connect("window-state-event", self.window_state_event_cb)

        # Deferred setup
        self._done_realize = False
        self.connect("realize", self._realize_cb)

        self.app.filehandler.current_file_observers.append(self.update_title)

        # Park the focus on the main tdw rather than on the toolbar. Default
        # activation doesn't really mean much for MyPaint's main window, so
        # it's safe to do this and it looks better.
        #self.main_widget.set_can_default(True)
        #self.main_widget.set_can_focus(True)
        #self.main_widget.grab_focus()


    def _realize_cb(self, drawwindow):
        # Deferred setup: anything that needs to be done when self.app is fully
        # initialized.
        if self._done_realize:
            return
        self._done_realize = True

        doc = self.app.doc
        tdw = doc.tdw
        assert tdw is self.app.builder.get_object("app_canvas")
        tdw.display_overlays.append(FrameOverlay(doc))
        self.update_overlays()
        self._init_actions()
        kbm = self.app.kbm
        kbm.add_extra_key('Menu', 'ShowPopupMenu')
        kbm.add_extra_key('Tab', 'FullscreenAutohide')
        self._init_stategroups()

        self._init_menubar()
        self._init_toolbar()
        topbar = self.app.builder.get_object("app_topbar")
        topbar.menubar = self.menubar
        topbar.toolbar = self.toolbar

        # Workspace setup
        ws = self.app.workspace
        ws.tool_widget_shown += self.app_workspace_tool_widget_shown_cb
        ws.tool_widget_hidden += self.app_workspace_tool_widget_hidden_cb

        # Footer bar updates
        self.app.brush.observers.append(self._update_footer_color_widgets)
        btn = self.app.builder.get_object("footer_mode_options_button")
        action = self.app.find_action("ModeOptionsTool")
        btn.set_related_action(action)
        tdw.transformation_updated += self._update_footer_scale_label
        doc.modes.observers.append(self._update_status_bar_mode_widgets)
        context_id = self.app.statusbar.get_context_id("active-mode")
        self._active_mode_context_id = context_id
        self._update_status_bar_mode_widgets(doc.modes.top)
        mode_img = self.app.builder.get_object("app_current_mode_icon")
        mode_img.connect("query-tooltip", self._mode_icon_query_tooltip_cb)
        mode_img.set_has_tooltip(True)


    def _init_actions(self):
        # Actions are defined in mypaint.xml: all we need to do here is connect
        # some extra state management.

        ag = self.action_group = self.app.builder.get_object("WindowActions")
        self.update_fullscreen_action()

        # Set initial state from user prefs
        ag.get_action("ToggleScaleFeedback").set_active(
                self.app.preferences.get("ui.feedback.scale", False))
        ag.get_action("ToggleLastPosFeedback").set_active(
                self.app.preferences.get("ui.feedback.last_pos", False))
        ag.get_action("ToggleSymmetryFeedback").set_active(
                self.app.preferences.get("ui.feedback.symmetry", False))

        # Keyboard handling
        for action in self.action_group.list_actions():
            self.app.kbm.takeover_action(action)

        # Brush chooser
        self._brush_chooser_dialog = None


    def _init_stategroups(self):
        sg = stategroup.StateGroup()
        p2s = sg.create_popup_state
        changer_crossed_bowl = p2s(colorselectionwindow.ColorChangerCrossedBowlPopup(self.app))
        changer_wash = p2s(colorselectionwindow.ColorChangerWashPopup(self.app))
        ring = p2s(colorselectionwindow.ColorRingPopup(self.app))
        hist = p2s(historypopup.HistoryPopup(self.app, self.app.doc.model))

        self.popup_states = {
            'ColorChangerCrossedBowlPopup': changer_crossed_bowl,
            'ColorChangerWashPopup': changer_wash,
            'ColorRingPopup': ring,
            'ColorHistoryPopup': hist,
            }

        # not sure how useful this is; we can't cycle at the moment
        changer_crossed_bowl.next_state = ring
        ring.next_state = changer_wash
        changer_wash.next_state = ring

        changer_wash.autoleave_timeout = None
        changer_crossed_bowl.autoleave_timeout = None
        ring.autoleave_timeout = None

        hist.autoleave_timeout = 0.600
        self.history_popup_state = hist

        for action_name, popup_state in self.popup_states.iteritems():
            label = self.app.find_action(action_name).get_label()
            popup_state.label = label


    def _init_menubar(self):
        # Load Menubar, duplicate into self.popupmenu
        menupath = os.path.join(self.app.datapath, 'gui/menu.xml')
        menubar_xml = open(menupath).read()
        self.app.ui_manager.add_ui_from_string(menubar_xml)
        self.popupmenu = self._clone_menu(menubar_xml, 'PopupMenu', self.app.doc.tdw)
        self.menubar = self.app.ui_manager.get_widget('/Menubar')


    def _init_toolbar(self):
        self.toolbar_manager = toolbar.ToolbarManager(self)
        self.toolbar = self.toolbar_manager.toolbar1

    def _clone_menu(self, xml, name, owner=None):
        """Menu duplicator

        Hopefully temporary hack for converting UIManager XML describing the
        main menubar into a rebindable popup menu. UIManager by itself doesn't
        let you do this, by design, but we need a bigger menu than the little
        things it allows you to build.
        """
        ui_elt = ET.fromstring(xml)
        rootmenu_elt = ui_elt.find("menubar")
        rootmenu_elt.attrib["name"] = name
        xml = ET.tostring(ui_elt)
        self.app.ui_manager.add_ui_from_string(xml)
        tmp_menubar = self.app.ui_manager.get_widget('/' + name)
        popupmenu = gtk.Menu()
        for item in tmp_menubar.get_children():
            tmp_menubar.remove(item)
            popupmenu.append(item)
        if owner is not None:
            popupmenu.attach_to_widget(owner, None)
        popupmenu.set_title("MyPaint")
        popupmenu.connect("selection-done", self.popupmenu_done_cb)
        popupmenu.connect("deactivate", self.popupmenu_done_cb)
        popupmenu.connect("cancel", self.popupmenu_done_cb)
        self.popupmenu_last_active = None
        return popupmenu


    def update_title(self, filename):
        if filename:
            self.set_title("MyPaint - %s" % os.path.basename(filename))
        else:
            self.set_title("MyPaint")

    # INPUT EVENT HANDLING
    def drag_data_received(self, widget, context, x, y, selection, info, t):
        if info == 1:
            if selection.data:
                uri = selection.data.split("\r\n")[0]
                fn = helpers.uri2filename(uri)
                if os.path.exists(fn):
                    if self.app.filehandler.confirm_destructive_action():
                        self.app.filehandler.open_file(fn)
        elif info == 2: # color
            color = RGBColor.new_from_drag_data(selection.data)
            self.app.brush_color_manager.set_color(color)
            self.app.brush_color_manager.push_history(color)
            # Don't popup the color history for now, as I haven't managed
            # to get it to cooperate.

    def print_memory_leak_cb(self, action):
        helpers.record_memory_leak_status(print_diff = True)

    def run_garbage_collector_cb(self, action):
        helpers.run_garbage_collector()

    def start_profiling_cb(self, action):
        if getattr(self, 'profiler_active', False):
            self.profiler_active = False
            return

        def doit():
            import cProfile
            profile = cProfile.Profile()

            self.profiler_active = True
            logger.info('--- GUI Profiling starts ---')
            while self.profiler_active:
                profile.runcall(gtk.main_iteration_do, False)
                if not gtk.events_pending():
                    time.sleep(0.050) # ugly trick to remove "user does nothing" from profile
            logger.info('--- GUI Profiling ends ---')

            profile.dump_stats('profile_fromgui.pstats')
            logger.debug('profile written to mypaint_profile.pstats')
            if os.path.exists("profile_fromgui.png"):
                os.unlink("profile_fromgui.png")
            os.system('gprof2dot.py -f pstats profile_fromgui.pstats | dot -Tpng -o profile_fromgui.png')
            if os.path.exists("profile_fromgui.png"):
                os.system('xdg-open profile_fromgui.png &')

        gobject.idle_add(doit)

    def _get_active_doc(self):
        # Determines which is the active doc for the purposes of keyboard
        # event dispatch.
        tdw_class = self.app.scratchpad_doc.tdw.__class__
        tdw = tdw_class.get_active_tdw()
        if tdw is not None:
            if tdw is self.app.scratchpad_doc.tdw:
                return (self.app.scratchpad_doc, tdw)
            elif tdw is self.app.doc.tdw:
                return (self.app.doc, tdw)
        return (None, None)

    def key_press_event_cb(self, win, event):
        # Process keyboard events
        target_doc, target_tdw = self._get_active_doc()
        if target_doc is None:
            return False
        # Unfullscreen
        if self.is_fullscreen and event.keyval == keysyms.Escape:
            gobject.idle_add(self.unfullscreen)
        # Forward the keypress to the active doc's active InteractionMode.
        return target_doc.modes.top.key_press_cb(win, target_tdw, event)


    def key_release_event_cb(self, win, event):
        # Process key-release events
        target_doc, target_tdw = self._get_active_doc()
        if target_doc is None:
            return False
        # Forward the event (see above)
        return target_doc.modes.top.key_release_cb(win, target_tdw, event)


    # Window handling
    def toggle_window_cb(self, action):
        """Handles a variety of window-toggling GtkActions.

        Handled here:

        * Workspace-managed tool widgets which require no constructor args.
        * Regular app subwindows, exposed via its get_subwindow() method.

        """
        action_name = action.get_name()
        if action_name.endswith("Tool") or action_name.endswith("Panel"):
            gtype_name = "MyPaint%s" % (action.get_name(),)
            workspace = self.app.workspace
            showing = workspace.get_tool_widget_showing(gtype_name, [])
            active = action.get_active()
            if active and not showing:
                workspace.show_tool_widget(gtype_name, [])
            elif showing and not active:
                workspace.hide_tool_widget(gtype_name, [])
        elif self.app.has_subwindow(action_name):
            window = self.app.get_subwindow(action_name)
            active = action.get_active()
            visible = window.get_visible()
            if active:
                if not visible:
                    window.show_all()
                window.present()
            elif visible:
                if not active:
                    window.hide()
        else:
            logger.warning("unknown window or tool %r" % (action_name,))


    def app_workspace_tool_widget_shown_cb(self, ws, widget):
        gtype_name = widget.__gtype_name__
        assert gtype_name.startswith("MyPaint")
        action_name = gtype_name.replace("MyPaint", "", 1)
        action = self.app.builder.get_object(action_name)
        if action and not action.get_active():
            action.set_active(True)


    def app_workspace_tool_widget_hidden_cb(self, ws, widget):
        gtype_name = widget.__gtype_name__
        assert gtype_name.startswith("MyPaint")
        action_name = gtype_name.replace("MyPaint", "", 1)
        action = self.app.builder.get_object(action_name)
        if action and action.get_active():
            action.set_active(False)


    # Feedback and overlays
    # It's not intended that all categories of feedback will use overlays, but
    # they currently all do. This may change now we have a conventional
    # statusbar for textual types of feedback.

    def toggle_scale_feedback_cb(self, action):
        self.app.preferences['ui.feedback.scale'] = action.get_active()
        self.update_overlays()

    def toggle_last_pos_feedback_cb(self, action):
        self.app.preferences['ui.feedback.last_pos'] = action.get_active()
        self.update_overlays()

    def toggle_symmetry_feedback_cb(self, action):
        self.app.preferences['ui.feedback.symmetry'] = action.get_active()
        self.update_overlays()

    def update_overlays(self):
        # Updates the list of overlays on the main doc's TDW to match the prefs
        doc = self.app.doc
        disp_overlays = [
            ('ui.feedback.scale', ScaleOverlay),
            ('ui.feedback.last_pos', LastPaintPosOverlay),
            ('ui.feedback.symmetry', SymmetryOverlay),
            ]
        overlays_changed = False
        for key, class_ in disp_overlays:
            current_instance = None
            for ov in doc.tdw.display_overlays:
                if isinstance(ov, class_):
                    current_instance = ov
            active = self.app.preferences.get(key, False)
            if active and not current_instance:
                doc.tdw.display_overlays.append(class_(doc))
                overlays_changed = True
            elif current_instance and not active:
                doc.tdw.display_overlays.remove(current_instance)
                overlays_changed = True
        if overlays_changed:
            doc.tdw.queue_draw()


    def popup_cb(self, action):
        state = self.popup_states[action.get_name()]
        state.activate(action)


    def brush_chooser_popup_cb(self, action):
        dialog = self._brush_chooser_dialog
        if dialog is None:
            dialog = dialogs.BrushChooserDialog(self.app)
            dialog.connect("response", self._brush_chooser_dialog_response_cb)
            self._brush_chooser_dialog = dialog
        if not dialog.get_visible():
            dialog.show_all()
            dialog.present()
        else:
            dialog.response(gtk.RESPONSE_CANCEL)


    def _brush_chooser_dialog_response_cb(self, dialog, response_id):
        dialog.hide()


    def color_details_dialog_cb(self, action):
        mgr = self.app.brush_color_manager
        new_col = RGBColor.new_from_dialog(
          title=_("Set current color"),
          color=mgr.get_color(),
          previous_color=mgr.get_previous_color(),
          parent=self)
        if new_col is not None:
            mgr.set_color(new_col)


    # Show Subwindows

    def fullscreen_autohide_toggled_cb(self, action):
        workspace = self.app.workspace
        workspace.autohide_enabled = action.get_active()


    # Fullscreen mode
    # This implementation requires an ICCCM and EWMH-compliant window manager
    # which supports the _NET_WM_STATE_FULLSCREEN hint. There are several
    # available.

    def fullscreen_cb(self, *junk):
        if not self.is_fullscreen:
            self.fullscreen()
        else:
            self.unfullscreen()

    def window_state_event_cb(self, widget, event):
        # Respond to changes of the fullscreen state only
        if not event.changed_mask & gdk.WINDOW_STATE_FULLSCREEN:
            return
        self.is_fullscreen = event.new_window_state & gdk.WINDOW_STATE_FULLSCREEN
        self.update_fullscreen_action()

    def update_fullscreen_action(self):
        action = self.action_group.get_action("Fullscreen")
        if self.is_fullscreen:
            action.set_icon_name("mypaint-unfullscreen-symbolic")
            action.set_tooltip(_("Leave Fullscreen Mode"))
            action.set_label(_("Leave Fullscreen"))
        else:
            action.set_icon_name("mypaint-fullscreen-symbolic")
            action.set_tooltip(_("Enter Fullscreen Mode"))
            action.set_label(_("Fullscreen"))

    def popupmenu_show_cb(self, action):
        self.show_popupmenu()

    def show_popupmenu(self, event=None):
        self.menubar.set_sensitive(False)   # excessive feedback?
        button = 1
        time = 0
        if event is not None:
            if event.type == gdk.BUTTON_PRESS:
                button = event.button
                time = event.time
        # GTK3: arguments have a different order, and "data" is required.
        # GTK3: Use keyword arguments for max compatibility.
        self.popupmenu.popup(parent_menu_shell=None, parent_menu_item=None,
                             func=None, button=button, activate_time=time,
                             data=None)
        if event is None:
            # We're responding to an Action, most probably the menu key.
            # Open out the last highlighted menu to speed key navigation up.
            if self.popupmenu_last_active is None:
                self.popupmenu.select_first(True) # one less keypress
            else:
                self.popupmenu.select_item(self.popupmenu_last_active)

    def popupmenu_done_cb(self, *a, **kw):
        # Not sure if we need to bother with this level of feedback,
        # but it actually looks quite nice to see one menu taking over
        # the other. Makes it clear that the popups are the same thing as
        # the full menu, maybe.
        self.menubar.set_sensitive(True)
        self.popupmenu_last_active = self.popupmenu.get_active()

    # BEGIN -- Scratchpad menu options
    def save_scratchpad_as_default_cb(self, action):
        self.app.filehandler.save_scratchpad(self.app.filehandler.get_scratchpad_default(), export = True)

    def clear_default_scratchpad_cb(self, action):
        self.app.filehandler.delete_default_scratchpad()

    # Unneeded since 'Save blank canvas' bug has been addressed.
    #def clear_autosave_scratchpad_cb(self, action):
    #    self.app.filehandler.delete_autosave_scratchpad()

    def new_scratchpad_cb(self, action):
        if os.path.isfile(self.app.filehandler.get_scratchpad_default()):
            self.app.filehandler.open_scratchpad(self.app.filehandler.get_scratchpad_default())
        else:
            self.app.scratchpad_doc.model.clear()
            # With no default - adopt the currently chosen background
            bg_layer = self.app.doc.model.layer_stack.background_layer
            if self.app.scratchpad_doc:
                self.app.scratchpad_doc.model.set_background(bg_layer)

        self.app.scratchpad_filename = self.app.preferences['scratchpad.last_opened'] = self.app.filehandler.get_scratchpad_autosave()

    def load_scratchpad_cb(self, action):
        if self.app.scratchpad_filename:
            self.save_current_scratchpad_cb(action)
            current_pad = self.app.scratchpad_filename
        else:
            current_pad = self.app.filehandler.get_scratchpad_autosave()
        self.app.filehandler.open_scratchpad_dialog()
        # Check to see if a file has been opened outside of the scratchpad directory
        if not os.path.abspath(self.app.scratchpad_filename).startswith(os.path.abspath(self.app.filehandler.get_scratchpad_prefix())):
            # file is NOT within the scratchpad directory - load copy as current scratchpad
            self.app.scratchpad_filename = self.app.preferences['scratchpad.last_opened'] = current_pad

    def save_as_scratchpad_cb(self, action):
        self.app.filehandler.save_scratchpad_as_dialog()

    def revert_current_scratchpad_cb(self, action):
        filename = self.app.scratchpad_filename
        if os.path.isfile(filename):
            self.app.filehandler.open_scratchpad(filename)
            logger.info("Reverted scratchpad to %s" % (filename,))
        else:
            logger.warning("No file to revert to yet.")

    def save_current_scratchpad_cb(self, action):
        self.app.filehandler.save_scratchpad(self.app.scratchpad_filename)

    def scratchpad_copy_background_cb(self, action):
        bg_layer = self.app.doc.model.layer_stack.background_layer
        if self.app.scratchpad_doc:
            self.app.scratchpad_doc.model.set_background(bg_layer)

    def draw_sat_spectrum_cb(self, action):
        g = GimpPalette()
        hsv = self.app.brush.get_color_hsv()
        g.append_sat_spectrum(hsv)
        grid_size = 30.0
        off_x = off_y = grid_size / 2.0
        column_limit = 8
        draw_palette(self.app, g, self.app.scratchpad_doc, columns=column_limit, grid_size=grid_size)

    # END -- Scratchpad menu options


    def palette_next_cb(self, action):
        mgr = self.app.brush_color_manager
        color = mgr.get_color()
        newcolor = mgr.palette.move_match_position(1, mgr.get_color())
        if newcolor:
            mgr.set_color(newcolor)
        # Show the palette panel if hidden
        workspace = self.app.workspace
        workspace.show_tool_widget("MyPaintPaletteTool", [])


    def palette_prev_cb(self, action):
        mgr = self.app.brush_color_manager
        color = mgr.get_color()
        newcolor = mgr.palette.move_match_position(-1, mgr.get_color())
        if newcolor:
            mgr.set_color(newcolor)
        # Show the palette panel if hidden
        workspace = self.app.workspace
        workspace.show_tool_widget("MyPaintPaletteTool", [])


    def palette_add_current_color_cb(self, *args, **kwargs):
        """Append the current color to the palette (action or clicked cb)"""
        mgr = self.app.brush_color_manager
        color = mgr.get_color()
        mgr.palette.append(color, name=None, unique=True, match=True)
        # Show the palette panel if hidden
        workspace = self.app.workspace
        workspace.show_tool_widget("MyPaintPaletteTool", [])


    def quit_cb(self, *junk):
        self.app.doc.model.flush_updates()
        self.app.save_gui_config() # FIXME: should do this periodically, not only on quit

        if not self.app.filehandler.confirm_destructive_action(title=_('Quit'), question=_('Really Quit?')):
            return True

        self.app.doc.model.cleanup()
        gtk.main_quit()
        return False

    def download_brush_pack_cb(self, *junk):
        url = BRUSHPACK_URI
        logger.info('Opening URL %r in web browser' % (url,))
        webbrowser.open(url)

    def import_brush_pack_cb(self, *junk):
        format_id, filename = dialogs.open_dialog(_("Import brush package..."), self,
                                 [(_("MyPaint brush package (*.zip)"), "*.zip")])
        if not filename:
            return
        imported = self.app.brushmanager.import_brushpack(filename,  self)
        logger.info("Imported brush groups %r", imported)
        workspace = self.app.workspace
        for groupname in imported:
            workspace.show_tool_widget("MyPaintBrushGroupTool", (groupname,))

    # INFORMATION
    # TODO: Move into dialogs.py?
    def about_cb(self, action):
        d = gtk.AboutDialog()
        d.set_transient_for(self)
        d.set_program_name("MyPaint")
        d.set_version(self.app.version)
        d.set_copyright(_("Copyright (C) 2005-2012\nMartin Renold and the MyPaint Development Team"))
        d.set_website("http://mypaint.info/")
        d.set_logo(self.app.pixmaps.mypaint_logo)
        d.set_license(
            _(u"This program is free software; you can redistribute it and/or modify "
              u"it under the terms of the GNU General Public License as published by "
              u"the Free Software Foundation; either version 2 of the License, or "
              u"(at your option) any later version.\n"
              u"\n"
              u"This program is distributed in the hope that it will be useful, "
              u"but WITHOUT ANY WARRANTY. See the COPYING file for more details.")
            )
        d.set_wrap_license(True)
        d.set_authors([
            # (in order of appearance)
            u"Martin Renold (%s)" % _('programming'),
            u"Yves Combe (%s)" % _('portability'),
            u"Popolon (%s)" % _('programming'),
            u"Clement Skau (%s)" % _('programming'),
            u"Jon Nordby (%s)" % _('programming'),
            u"Álinson Santos (%s)" % _('programming'),
            u"Tumagonx (%s)" % _('portability'),
            u"Ilya Portnov (%s)" % _('programming'),
            u"Jonas Wagner (%s)" % _('programming'),
            u"Luka Čehovin (%s)" % _('programming'),
            u"Andrew Chadwick (%s)" % _('programming'),
            u"Till Hartmann (%s)" % _('programming'),
            u'David Grundberg (%s)' % _('programming'),
            u"Krzysztof Pasek (%s)" % _('programming'),
            u"Ben O'Steen (%s)" % _('programming'),
            u"Ferry Jérémie (%s)" % _('programming'),
            u"しげっち 'sigetch' (%s)" % _('programming'),
            u"Richard Jones (%s)" % _('programming'),
            u"David Gowers (%s)" % _('programming'),
            ])
        d.set_artists([
            u"Artis Rozentāls (%s)" % _('brushes'),
            u"Popolon (%s)" % _('brushes'),
            u"Marcelo 'Tanda' Cerviño (%s)" % _('patterns, brushes'),
            u"David Revoy (%s)" % _('brushes, tool icons'),
            u"Ramón Miranda (%s)" % _('brushes, patterns'),
            u"Enrico Guarnieri 'Ico_dY' (%s)" % _('brushes'),
            u'Sebastian Kraft (%s)' % _('desktop icon'),
            u"Nicola Lunghi (%s)" % _('patterns'),
            u"Toni Kasurinen (%s)" % _('brushes'),
            u"Сан Саныч 'MrMamurk' (%s)" % _('patterns'),
            u"Andrew Chadwick (%s)" % _('tool icons'),
            u"Ben O'Steen (%s)" % _('tool icons'),
            ])
        d.set_translator_credits(_("translator-credits"));

        d.run()
        d.destroy()

    def show_infodialog_cb(self, action):
        text = {
        'ShortcutHelp':
                _("Move your mouse over a menu entry, then press the key to assign."),
        'ContextHelp':
                _("Brush shortcut keys are used to quickly save/restore brush "
                 "settings. You can paint with one hand and change brushes with "
                 "the other, even in mid-stroke."
                 "\n\n"
                 "There are 10 persistent memory slots available."),
        'Docu':
                _("There is a tutorial available on the MyPaint homepage. It "
                 "explains some features which are hard to discover yourself."
                 "\n\n"
                 "Comments about the brush settings (opaque, hardness, etc.) and "
                 "inputs (pressure, speed, etc.) are available as tooltips. "
                 "Put your mouse over a label to see them. "
                 "\n"),
        }
        self.app.message_dialog(text[action.get_name()])


    ## Footer bar stuff

    def _update_footer_color_widgets(self, settings):
        """Updates the footer bar color info when the brush color changes."""
        if not settings.intersection(('color_h', 'color_s', 'color_v')):
            return
        bm_btn_name = "footer_bookmark_current_color_button"
        bm_btn = self.app.builder.get_object(bm_btn_name)
        brush_color = HSVColor(*self.app.brush.get_color_hsv())
        palette = self.app.brush_color_manager.palette
        bm_btn.set_sensitive(brush_color not in palette)

    def _update_footer_scale_label(self, renderer):
        """Updates the footer's scale label when the transformation changes"""
        label = self.app.builder.get_object("app_canvas_scale_label")
        scale = renderer.scale * 100.0
        rotation = (renderer.rotation / (2*math.pi)) % 1.0
        if rotation > 0.5:
            rotation -= 1.0
        rotation *= 360.0
        try:
            template = label.__template
        except AttributeError:
            template = label.get_label()
            label.__template = template
        params = { "scale": scale,
                   "rotation": rotation }
        label.set_text(template.format(**params))

    def _update_status_bar_mode_widgets(self, mode):
        """Updates widgets on the status bar that reflect the current mode"""
        # Update the status bar
        statusbar = self.app.statusbar
        context_id = self._active_mode_context_id
        statusbar.pop(context_id)
        statusbar_msg = u"{usage!s}".format(name=mode.get_name(),
                                            usage=mode.get_usage())
        statusbar.push(context_id, statusbar_msg)
        # Icon
        icon_name = mode.get_icon_name()
        icon_size = gtk.ICON_SIZE_SMALL_TOOLBAR
        mode_img = self.app.builder.get_object("app_current_mode_icon")
        if not icon_name:
            icon_name = "missing-image"
        mode_img.set_from_icon_name(icon_name, icon_size)

    def _mode_icon_query_tooltip_cb(self, widget, x, y, kbmode, tooltip):
        mode = self.app.doc.modes.top
        icon_name = mode.get_icon_name()
        if not icon_name:
            icon_name = "missing-image"
        icon_size = gtk.ICON_SIZE_DIALOG
        tooltip.set_icon_from_icon_name(icon_name, icon_size)
        description = None
        action = mode.get_action()
        if action:
            description = action.get_tooltip()
        if not description:
            description = mode.get_usage()
        params = { "name": helpers.escape(mode.get_name()),
                   "description": helpers.escape(description) }
        markup = self._MODE_ICON_TEMPLATE.format(**params)
        tooltip.set_markup(markup)
        return True
