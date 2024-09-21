# -*- coding: utf-8 -*-
#
# This file is part of MyPaint.
# Copyright (C) 2007-2019 by the MyPaint Development Team.
# Copyright (C) 2007-2014 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Main drawing window.

Painting is done in tileddrawwidget.py.
"""

## Imports

from __future__ import division, print_function

import os
import os.path
import webbrowser
from warnings import warn
import logging
import math
import xml.etree.ElementTree as ET

from lib.gibindings import Gtk
from lib.gibindings import Gdk

from . import compatibility
from . import historypopup
from . import stategroup
from . import colorpicker  # noqa: F401 (registration of GObject classes)
from . import windowing  # noqa: F401 (registration of GObject classes)
from . import toolbar
from . import dialogs
from . import layermodes  # noqa: F401 (registration of GObject classes)
from . import quickchoice
import gui.viewmanip  # noqa: F401 (registration of GObject classes)
import gui.layermanip  # noqa: F401 (registration of GObject classes)
import gui.brushmanip  # noqa: F401
from lib.color import HSVColor
from . import uicolor
import gui.picker
import gui.footer
from . import brushselectionwindow  # noqa: F401 (registration)
from .overlays import LastPaintPosOverlay
from .overlays import ScaleOverlay
from .framewindow import FrameOverlay
from .symmetry import SymmetryOverlay
import gui.tileddrawwidget
import gui.displayfilter
import gui.meta
import lib.xml
import lib.glib
from lib.gettext import gettext as _
from lib.gettext import C_

logger = logging.getLogger(__name__)


## Module constants

BRUSHPACK_URI = 'https://github.com/mypaint/mypaint/wiki/Brush-Packages'


## Class definitions

class DrawWindow (Gtk.Window):
    """Main drawing window"""

    ## Class configuration

    __gtype_name__ = 'MyPaintDrawWindow'

    _MODE_ICON_TEMPLATE = "<b>{name}</b>\n{description}"

    #: Constructor callables and canned args for named quick chooser
    #: instances. Used by _get_quick_chooser().
    _QUICK_CHOOSER_CONSTRUCT_INFO = {
        "BrushChooserPopup": (
            quickchoice.BrushChooserPopup, [],
        ),
        "ColorChooserPopup": (
            quickchoice.ColorChooserPopup, [],
        ),
        "ColorChooserPopupFastSubset": (
            quickchoice.ColorChooserPopup, ["fast_subset", True],
        ),
    }

    ## Initialization and lifecycle

    def __init__(self):
        super(DrawWindow, self).__init__()

        import gui.application
        app = gui.application.get_app()
        self.app = app
        self.app.kbm.add_window(self)

        # Window handling
        self._updating_toggled_item = False
        self.is_fullscreen = False

        # Enable drag & drop
        drag_targets = [
            Gtk.TargetEntry.new("text/uri-list", 0, 1),
            Gtk.TargetEntry.new("application/x-color", 0, 2)
        ]
        drag_flags = (
            Gtk.DestDefaults.MOTION |
            Gtk.DestDefaults.HIGHLIGHT |
            Gtk.DestDefaults.DROP)
        drag_actions = Gdk.DragAction.DEFAULT | Gdk.DragAction.COPY
        self.drag_dest_set(drag_flags, drag_targets, drag_actions)

        # Connect events
        self.connect('delete-event', self.quit_cb)
        self.connect("drag-data-received", self._drag_data_received_cb)
        self.connect("window-state-event", self.window_state_event_cb)
        self.connect("button-press-event", self._button_press_cb)

        # Deferred setup
        self._done_realize = False
        self.connect("realize", self._realize_cb)

        self.app.filehandler.current_file_observers.append(self.update_title)

        # Named quick chooser instances
        self._quick_choosers = {}

        # Park the focus on the main tdw rather than on the toolbar. Default
        # activation doesn't really mean much for MyPaint's main window, so
        # it's safe to do this and it looks better.
        #   self.main_widget.set_can_default(True)
        #   self.main_widget.set_can_focus(True)
        #   self.main_widget.grab_focus()

    def _button_press_cb(self, window, event):
        windowing.clear_focus(window)

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
        tdw.display_overlays.append(SymmetryOverlay(doc))
        self.update_overlays()
        self._init_actions()
        kbm = self.app.kbm
        kbm.add_extra_key('Menu', 'ShowPopupMenu')
        kbm.add_extra_key('Tab', 'FullscreenAutohide')
        self._init_stategroups()

        self._init_menubar()
        self._init_toolbars()
        topbar = self.app.builder.get_object("app_topbar")
        topbar.menubar = self.menubar
        topbar.toolbar1 = self._toolbar1
        topbar.toolbar2 = self._toolbar2

        # Workspace setup
        ws = self.app.workspace
        ws.tool_widget_added += self.app_workspace_tool_widget_added_cb
        ws.tool_widget_removed += self.app_workspace_tool_widget_removed_cb

        # Footer bar updates
        self.app.brush.observers.append(self._update_footer_color_widgets)
        tdw.transformation_updated += self._update_footer_scale_label
        doc.modes.changed += self._modestack_changed_cb
        context_id = self.app.statusbar.get_context_id("active-mode")
        self._active_mode_context_id = context_id
        self._update_status_bar_mode_widgets(doc.modes.top)
        mode_img = self.app.builder.get_object("app_current_mode_icon")
        mode_img.connect("query-tooltip", self._mode_icon_query_tooltip_cb)
        mode_img.set_has_tooltip(True)

        # Update picker action sensitivity
        layerstack = doc.model.layer_stack
        layerstack.layer_inserted += self._update_layer_pick_action
        layerstack.layer_deleted += self._update_layer_pick_action

    def _init_actions(self):
        # Actions are defined in resources.xml.
        # all we need to do here is connect some extra state management.

        ag = self.action_group = self.app.builder.get_object("WindowActions")
        self.update_fullscreen_action()

        # Set initial state from user prefs
        ag.get_action("ToggleScaleFeedback").set_active(
            self.app.preferences.get("ui.feedback.scale", False))
        ag.get_action("ToggleLastPosFeedback").set_active(
            self.app.preferences.get("ui.feedback.last_pos", False))

        # Keyboard handling
        for action in self.action_group.list_actions():
            self.app.kbm.takeover_action(action)

    def _init_stategroups(self):
        sg = stategroup.StateGroup()
        p2s = sg.create_popup_state
        hist = p2s(historypopup.HistoryPopup(self.app, self.app.doc.model))

        self.popup_states = {
            'ColorHistoryPopup': hist,
        }

        hist.autoleave_timeout = 0.600
        self.history_popup_state = hist

        for action_name, popup_state in self.popup_states.items():
            label = self.app.find_action(action_name).get_label()
            popup_state.label = label

    def _init_menubar(self):
        # Load Menubar, duplicate into self.popupmenu
        ui_dir = os.path.dirname(os.path.abspath(__file__))
        menupath = os.path.join(ui_dir, 'menu.xml')
        with open(menupath) as fp:
            menubar_xml = fp.read()
        self.app.ui_manager.add_ui_from_string(menubar_xml)
        self.popupmenu = self._clone_menu(
            menubar_xml,
            'PopupMenu',
            self.app.doc.tdw,
        )
        self.menubar = self.app.ui_manager.get_widget('/Menubar')

    def _init_toolbars(self):
        self._toolbar_manager = toolbar.ToolbarManager(self)
        self._toolbar1 = self._toolbar_manager.toolbar1
        self._toolbar2 = self._toolbar_manager.toolbar2

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
        xml = xml.decode("utf-8")
        self.app.ui_manager.add_ui_from_string(xml)
        tmp_menubar = self.app.ui_manager.get_widget('/' + name)
        popupmenu = Gtk.Menu()
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
            # TRANSLATORS: window title for use with a filename
            title_base = _("%s - MyPaint") % os.path.basename(filename)
        else:
            # TRANSLATORS: window title for use without a filename
            title_base = _("MyPaint")
        # Show whether legacy 1.x compatibility mode is active
        if self.app.compat_mode == compatibility.C1X:
            compat_str = " (%s)" % C_("Prefs Dialog|Compatibility", "1.x")
        else:
            compat_str = ""
        self.set_title(title_base + compat_str)

    def _drag_data_received_cb(self, widget, context, x, y, data, info, time):
        """Handles data being received"""
        rawdata = data.get_data()
        if not rawdata:
            return
        if info == 1:  # file uris
            # Perhaps these should be handled as layers instead now?
            # Though .ORA files should probably still replace the entire
            # working file.
            uri = lib.glib.filename_to_unicode(rawdata).split("\r\n")[0]
            file_path, _h = lib.glib.filename_from_uri(uri)
            if os.path.exists(file_path):
                ok_to_open = self.app.filehandler.confirm_destructive_action(
                    title = C_(
                        u'Open dragged file confirm dialog: title',
                        u"Open Dragged File?",
                    ),
                    confirm = C_(
                        u'Open dragged file confirm dialog: continue button',
                        u"_Open",
                    ),
                )
                if ok_to_open:
                    self.app.filehandler.open_file(file_path)
        elif info == 2:  # color
            color = uicolor.from_drag_data(rawdata)
            self.app.brush_color_manager.set_color(color)
            self.app.brush_color_manager.push_history(color)

    ## Window and dockpanel handling

    def reveal_dockpanel_cb(self, action):
        """Action callback: reveal a dockpanel in its current location.

        This adds the related dockpanel if it has not yet been added to
        the workspace. In fullscreen mode, the action also acts to show
        the sidebar or floating window which contains the dockpanel.
        It also brings its tab to the fore.

        The panel's name is parsed from the action name. An action name
        of 'RevealFooPanel' relates to a panel whose GType-system class
        name is "MyPaintFooPanel". Old-style "Tool" suffixes are
        supported too, but are deprecated.

        """
        action_name = action.get_name()
        if not action_name.startswith("Reveal"):
            raise ValueError("Action's name must start with 'Reveal'")
        type_name = action_name.replace("Reveal", "", 1)
        if not (type_name.endswith("Tool") or type_name.endswith("Panel")):
            raise ValueError("Action's name must end with 'Panel' or 'Tool'")
        gtype_name = "MyPaint" + type_name
        workspace = self.app.workspace
        workspace.reveal_tool_widget(gtype_name, [])

    def toggle_dockpanel_cb(self, action):
        """Action callback: add or remove a dockpanel from the UI."""
        action_name = action.get_name()
        type_name = action_name
        for prefix in ["Toggle"]:
            if type_name.startswith(prefix):
                type_name = type_name.replace(prefix, "", 1)
                break
        if not (type_name.endswith("Tool") or type_name.endswith("Panel")):
            raise ValueError("Action's name must end with 'Panel' or 'Tool'")
        gtype_name = "MyPaint" + type_name
        workspace = self.app.workspace
        added = workspace.get_tool_widget_added(gtype_name, [])
        active = action.get_active()
        if active and not added:
            workspace.add_tool_widget(gtype_name, [])
        elif added and not active:
            workspace.remove_tool_widget(gtype_name, [])

    def toggle_window_cb(self, action):
        """Handles a variety of window-toggling GtkActions.

        Handled here:

        * Workspace-managed dockpanels which require no constructor args.
        * Regular app subwindows, exposed via its get_subwindow() method.

        """
        action_name = action.get_name()
        if action_name.endswith("Tool") or action_name.endswith("Panel"):
            self.toggle_dockpanel_cb(action)
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

    def app_workspace_tool_widget_added_cb(self, ws, widget):
        gtype_name = widget.__gtype_name__
        self._set_tool_widget_related_toggleaction_active(gtype_name, True)

    def app_workspace_tool_widget_removed_cb(self, ws, widget):
        gtype_name = widget.__gtype_name__
        self._set_tool_widget_related_toggleaction_active(gtype_name, False)

    def _set_tool_widget_related_toggleaction_active(self, gtype_name, active):
        active = bool(active)
        assert gtype_name.startswith("MyPaint")
        for prefix in ("Toggle", ""):
            action_name = gtype_name.replace("MyPaint", prefix, 1)
            action = self.app.builder.get_object(action_name)
            if action and isinstance(action, Gtk.ToggleAction):
                if bool(action.get_active()) != active:
                    action.set_active(active)
                break

    ## Feedback and overlays

    # It's not intended that all categories of feedback will use
    # overlays, but they currently all do. This may change now we have a
    # conventional statusbar for textual types of feedback.

    def toggle_scale_feedback_cb(self, action):
        self.app.preferences['ui.feedback.scale'] = action.get_active()
        self.update_overlays()

    def toggle_last_pos_feedback_cb(self, action):
        self.app.preferences['ui.feedback.last_pos'] = action.get_active()
        self.update_overlays()

    def update_overlays(self):
        # Updates the list of overlays on the main doc's TDW to match the prefs
        doc = self.app.doc
        disp_overlays = [
            ('ui.feedback.scale', ScaleOverlay),
            ('ui.feedback.last_pos', LastPaintPosOverlay),
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

    ## Popup windows and dialogs

    def popup_cb(self, action):
        """Action callback: show a popup window (old mechanism)"""
        warn(
            "The old UI states mechanism is scheduled for replacement. "
            "Don't use this in new code.",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        state = self.popup_states[action.get_name()]
        state.activate(action)

    def _get_quick_chooser(self, name):
        """Get a named quick chooser instance (factory method)"""
        chooser = self._quick_choosers.get(name)
        if not chooser:
            ctor_info = self._QUICK_CHOOSER_CONSTRUCT_INFO.get(name)
            ctor, extra_args = ctor_info
            args = [self.app] + list(extra_args)
            chooser = ctor(*args)
            self._quick_choosers[name] = chooser
        return chooser

    def _popup_quick_chooser(self, name):
        """Pops up a named quick chooser instance, hides the others"""
        chooser = self._get_quick_chooser(name)
        if chooser.get_visible():
            chooser.advance()
            return
        for other_name in self._QUICK_CHOOSER_CONSTRUCT_INFO:
            if other_name == name:
                continue
            other_chooser = self._quick_choosers.get(other_name)
            if other_chooser and other_chooser.get_visible():
                other_chooser.hide()
        chooser.popup()

    def quick_chooser_popup_cb(self, action):
        """Action callback: show the named quick chooser (new system)"""
        chooser_name = action.get_name()
        self._popup_quick_chooser(chooser_name)

    @property
    def brush_chooser(self):
        """Property: the brush chooser"""
        return self._get_quick_chooser("BrushChooserPopup")

    @property
    def color_chooser(self):
        """Property: the primary color chooser"""
        return self._get_quick_chooser("ColorChooserPopup")

    def color_details_dialog_cb(self, action):
        mgr = self.app.brush_color_manager
        new_col = dialogs.ask_for_color(
            title=_("Set current color"),
            color=mgr.get_color(),
            previous_color=mgr.get_previous_color(),
            parent=self,
        )
        if new_col is not None:
            mgr.set_color(new_col)

    ## Subwindows

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
        if not event.changed_mask & Gdk.WindowState.FULLSCREEN:
            return
        self.is_fullscreen = (
            event.new_window_state & Gdk.WindowState.FULLSCREEN
        )
        self.update_fullscreen_action()
        # Reset all state for the top mode on the stack. Mainly for
        # freehand modes: https://github.com/mypaint/mypaint/issues/39
        mode = self.app.doc.modes.top
        mode.leave()
        mode.enter(doc=self.app.doc)
        # The alternative is to use checkpoint(), but if freehand were
        # to reinit its workarounds, that might cause glitches.

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
            if event.type == Gdk.EventType.BUTTON_PRESS:
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
                self.popupmenu.select_first(True)  # one less keypress
            else:
                self.popupmenu.select_item(self.popupmenu_last_active)

    def popupmenu_done_cb(self, *a, **kw):
        # Not sure if we need to bother with this level of feedback,
        # but it actually looks quite nice to see one menu taking over
        # the other. Makes it clear that the popups are the same thing as
        # the full menu, maybe.
        self.menubar.set_sensitive(True)
        self.popupmenu_last_active = self.popupmenu.get_active()

    ## Scratchpad menu options

    def save_scratchpad_as_default_cb(self, action):
        self.app.filehandler.save_scratchpad(
            self.app.filehandler.get_scratchpad_default(),
            export=True,
        )

    def clear_default_scratchpad_cb(self, action):
        self.app.filehandler.delete_default_scratchpad()

    def new_scratchpad_cb(self, action):
        app = self.app
        default_scratchpad_path = app.filehandler.get_scratchpad_default()
        if os.path.isfile(default_scratchpad_path):
            app.filehandler.open_scratchpad(default_scratchpad_path)
        else:
            scratchpad_model = app.scratchpad_doc.model
            scratchpad_model.clear()
            self._copy_main_background_to_scratchpad()
        scratchpad_path = app.filehandler.get_scratchpad_autosave()
        app.scratchpad_filename = scratchpad_path
        app.preferences['scratchpad.last_opened'] = scratchpad_path

    def load_scratchpad_cb(self, action):
        if self.app.scratchpad_filename:
            self.save_current_scratchpad_cb(action)
            current_pad = self.app.scratchpad_filename
        else:
            current_pad = self.app.filehandler.get_scratchpad_autosave()
        self.app.filehandler.open_scratchpad_dialog()

        # Check to see if a file has been opened
        # outside of the scratchpad directory
        path_abs = os.path.abspath(self.app.scratchpad_filename)
        pfx_abs = os.path.abspath(self.app.filehandler.get_scratchpad_prefix())
        if not path_abs.startswith(pfx_abs):
            # file is NOT within the scratchpad directory -
            # load copy as current scratchpad
            self.app.preferences['scratchpad.last_opened'] = current_pad
            self.app.scratchpad_filename = current_pad

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
        self._copy_main_background_to_scratchpad()

    def _copy_main_background_to_scratchpad(self):
        app = self.app
        if not app.scratchpad_doc:
            return
        main_model = app.doc.model
        main_bg_layer = main_model.layer_stack.background_layer
        scratchpad_model = app.scratchpad_doc.model
        scratchpad_model.layer_stack.set_background(main_bg_layer)

    ## Palette actions

    def palette_next_cb(self, action):
        mgr = self.app.brush_color_manager
        newcolor = mgr.palette.move_match_position(1, mgr.get_color())
        if newcolor:
            mgr.set_color(newcolor)
        # Show the palette panel if hidden
        workspace = self.app.workspace
        workspace.reveal_tool_widget("MyPaintPaletteTool", [])

    def palette_prev_cb(self, action):
        mgr = self.app.brush_color_manager
        newcolor = mgr.palette.move_match_position(-1, mgr.get_color())
        if newcolor:
            mgr.set_color(newcolor)
        # Show the palette panel if hidden
        workspace = self.app.workspace
        workspace.reveal_tool_widget("MyPaintPaletteTool", [])

    def palette_add_current_color_cb(self, *args, **kwargs):
        """Append the current color to the palette (action or clicked cb)"""
        mgr = self.app.brush_color_manager
        color = mgr.get_color()
        mgr.palette.append(color, name=None, unique=True, match=True)
        # Show the palette panel if hidden
        workspace = self.app.workspace
        workspace.reveal_tool_widget("MyPaintPaletteTool", [])

    ## Miscellaneous actions

    def quit_cb(self, *junk):
        self.app.doc.model.sync_pending_changes()
        self.app.save_gui_config()  # FIXME: should do this periodically
        ok_to_quit = self.app.filehandler.confirm_destructive_action(
            title = C_(
                "Quit confirm dialog: title",
                u"Really Quit?",
            ),
            confirm = C_(
                "Quit confirm dialog: continue button",
                u"_Quit",
            ),
        )
        if not ok_to_quit:
            return True

        self.app.doc.model.cleanup()
        self.app.profiler.cleanup()
        Gtk.main_quit()
        return False

    def download_brush_pack_cb(self, *junk):
        uri = BRUSHPACK_URI
        logger.info('Opening URI %r in web browser', uri)
        webbrowser.open(uri)

    def import_brush_pack_cb(self, *junk):
        format_id, filename = dialogs.open_dialog(
            _(u"Import brush package…"), self,
            [(_("MyPaint brush package (*.zip)"), "*.zip")]
        )
        if not filename:
            return
        imported = self.app.brushmanager.import_brushpack(filename, self)
        logger.info("Imported brush groups %r", imported)
        workspace = self.app.workspace
        for groupname in imported:
            workspace.reveal_tool_widget("MyPaintBrushGroupTool", (groupname,))

    ## Information dialogs

    # TODO: Move into dialogs.py?

    def about_cb(self, action):
        gui.meta.run_about_dialog(self, self.app)

    def show_online_help_cb(self, action):
        # The online help texts are migrating to the wiki for v1.2.x.
        wiki_base = "https://github.com/mypaint/mypaint/wiki/"
        action_name = action.get_name()
        # TODO: these page names should be localized.
        help_page = {
            "OnlineHelpIndex": "v1.2-User-Manual",
            "OnlineHelpBrushShortcutKeys": "v1.2-Brush-Shortcut-Keys",
        }.get(action_name)
        if help_page:
            help_uri = wiki_base + help_page
            logger.info('Opening URI %r in web browser', help_uri)
            webbrowser.open(help_uri)
        else:
            raise RuntimeError("Unknown online help %r" % action_name)

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
        params = {
            "scale": scale,
            "rotation": rotation
        }
        label.set_text(template.format(**params))

    def _modestack_changed_cb(self, modestack, old, new):
        self._update_status_bar_mode_widgets(new)

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
        icon_size = Gtk.IconSize.SMALL_TOOLBAR
        mode_img = self.app.builder.get_object("app_current_mode_icon")
        if not icon_name:
            icon_name = "missing-image"
        mode_img.set_from_icon_name(icon_name, icon_size)

    def _mode_icon_query_tooltip_cb(self, widget, x, y, kbmode, tooltip):
        mode = self.app.doc.modes.top
        icon_name = mode.get_icon_name()
        if not icon_name:
            icon_name = "missing-image"
        icon_size = Gtk.IconSize.DIALOG
        tooltip.set_icon_from_icon_name(icon_name, icon_size)
        description = None
        action = mode.get_action()
        if action:
            description = action.get_tooltip()
        if not description:
            description = mode.get_usage()
        params = {
            "name": lib.xml.escape(mode.get_name()),
            "description": lib.xml.escape(description)
        }
        markup = self._MODE_ICON_TEMPLATE.format(**params)
        tooltip.set_markup(markup)
        return True

    def _footer_color_details_button_realize_cb(self, button):
        action = self.app.find_action("ColorDetailsDialog")
        button.set_related_action(action)

    ## Footer picker buttons

    def _footer_context_picker_button_realize_cb(self, button):
        presenter = gui.picker.ButtonPresenter()
        presenter.set_button(button)
        presenter.set_picking_grab(self.app.context_grab)
        self._footer_context_picker_button_presenter = presenter

    def _footer_color_picker_button_realize_cb(self, button):
        presenter = gui.picker.ButtonPresenter()
        presenter.set_button(button)
        presenter.set_picking_grab(self.app.color_grab)
        self._footer_color_picker_button_presenter = presenter

    ## Footer indicator widgets

    def _footer_brush_indicator_drawingarea_realize_cb(self, drawarea):
        presenter = gui.footer.BrushIndicatorPresenter()
        presenter.set_drawing_area(drawarea)
        presenter.set_brush_manager(self.app.brushmanager)
        presenter.set_chooser(self.brush_chooser)
        self._footer_brush_indicator_presenter = presenter

    ## Picker actions (PickLayer, PickContext)

    # App-wide really, but they can be handled here sensibly while
    # there's only one window.

    def pick_context_cb(self, action):
        """Pick Context action: select layer and brush from stroke"""
        # Get the controller owning most recently moved painted to or
        # moved over view widget as its primary tdw.
        # That controller points at the doc we want to pick from.
        doc = self.app.doc.get_active_instance()
        if not doc:
            return
        x, y = doc.tdw.get_pointer_in_model_coordinates()
        doc.pick_context(x, y, action)

    def pick_layer_cb(self, action):
        """Pick Layer action: select the layer under the pointer"""
        doc = self.app.doc.get_active_instance()
        if not doc:
            return
        x, y = doc.tdw.get_pointer_in_model_coordinates()
        doc.pick_layer(x, y, action)

    def _update_layer_pick_action(self, layerstack, *_ignored):
        """Updates the Layer Picking action's sensitivity"""
        # PickContext is always sensitive, however
        pickable = len(layerstack) > 1
        self.app.find_action("PickLayer").set_sensitive(pickable)

    ## Display filter choice

    def _display_filter_radioaction_changed_cb(self, action, newaction):
        """Handle changes to the Display Filter radioaction set."""
        newaction_name = newaction.get_name()
        newfilter = {
            "DisplayFilterNone": None,
            "DisplayFilterLumaOnly": gui.displayfilter.luma_only,
            "DisplayFilterInvertColors": gui.displayfilter.invert_colors,
            "DisplayFilterSimDeuteranopia": gui.displayfilter.sim_deuteranopia,
            "DisplayFilterSimProtanopia": gui.displayfilter.sim_protanopia,
            "DisplayFilterSimTritanopia": gui.displayfilter.sim_tritanopia,
        }.get(newaction_name)
        for tdw in gui.tileddrawwidget.TiledDrawWidget.get_visible_tdws():
            if tdw.renderer.display_filter is newfilter:
                continue
            logger.debug("Updating display_filter on %r to %r", tdw, newfilter)
            tdw.renderer.display_filter = newfilter
            tdw.queue_draw()
