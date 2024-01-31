# This file is part of MyPaint.
# Copyright (C) 2012-2019 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Button press mapping."""

from __future__ import division, print_function
from gettext import gettext as _
import logging

from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import GObject
from lib.gibindings import Pango

import lib.xml
from . import widgets
from lib.pycompat import unicode

logger = logging.getLogger(__name__)


def button_press_name(button, mods):
    """Converts button number & modifier mask to a prefs-storable string.

    Analogous to `Gtk.accelerator_name()`.  Buttonpress names look similar to
    GDK accelerator names, for example ``<Control><Shift>Button2`` or
    ``<Primary><Alt>Button4`` for newer versions of GTK.  If the button is
    equal to zero (see `button_press_parse()`), `None` is returned.

    """
    button = int(button)
    mods = int(mods)
    if button <= 0:
        return None
    mods = Gdk.ModifierType(mods)
    modif_name = Gtk.accelerator_name(0, mods)
    return modif_name + "Button%d" % (button,)


def button_press_displayname(button, mods, shorten = False):
    """Converts a button number & modifier mask to a localized unicode string.
    """
    button = int(button)
    mods = int(mods)
    if button <= 0:
        return None
    mods = Gdk.ModifierType(mods)
    modif_label = Gtk.accelerator_get_label(0, mods)
    modif_label = unicode(modif_label)
    separator = ""
    if modif_label:
        separator = u"+"
    # TRANSLATORS: "Button" refers to a mouse button
    # TRANSLATORS: It is part of a button map label.
    mouse_button_label = _("Button")
    if shorten:
        # TRANSLATORS: abbreviated "Button <number>" for forms like "Alt+Btn1"
        mouse_button_label = _("Btn")
    return "{modifiers}{plus}{btn}{button_number}".format(
        modifiers=modif_label,
        plus=separator,
        btn=mouse_button_label,
        button_number=button,
    )


def button_press_parse(name):
    """Converts button press names to a button number & modifier mask.

    Analogous to `Gtk.accelerator_parse()`. This function parses the strings
    created by `button_press_name()`, and returns a 2-tuple containing the
    button number and modifier mask corresponding to `name`. If the parse
    fails, both values will be 0 (zero).

    """
    if name is None:
        return (0, 0)
    name = str(name)
    try:
        mods_s, button_s = name.split("Button", 1)
        if button_s == '':
            button = 0
        else:
            button = int(button_s)
    except ValueError:
        button = 0
        mods = Gdk.ModifierType(0)
    else:
        keyval_ignored, mods = Gtk.accelerator_parse(mods_s)
    return button, mods


def get_handler_object(app, action_name):
    """Find a (nominal) handler for a named buttonmap action.

    :param app: MyPaint application instance to use for the lookup
    :param action_name: machine-readable action name string.
    :rtype: tuple of the form (handler_type, handler_obj)

    Defined handler_type strings and their handler_objs are: "mode_class" (an
    instantiable InteractionMode class), "popup_state" (an activatable popup
    state), "gtk_action" (an activatable Gtk.Action), or "no_handler" (the
    value None).

    """
    from gui.mode import ModeRegistry, InteractionMode
    mode_class = ModeRegistry.get_mode_class(action_name)
    if mode_class is not None:
        assert issubclass(mode_class, InteractionMode)
        return ("mode_class", mode_class)
    elif action_name in app.drawWindow.popup_states:
        popup_state = app.drawWindow.popup_states[action_name]
        return ("popup_state", popup_state)
    else:
        action = app.find_action(action_name)
        if action is not None:
            return ("gtk_action", action)
        else:
            return ("no_handler", None)


class ButtonMapping (object):
    """Button mapping table.

    An instance resides in the application, and is updated by the preferences
    window.

    """

    def __init__(self):
        super(ButtonMapping, self).__init__()
        self._mapping = {}
        self._modifiers = []

    def update(self, mapping):
        """Updates from a prefs sub-hash.

        :param mapping: dict of button_press_name()s to action names.
           A reference is not maintained.

        """
        self._mapping = {}
        self._modifiers = []
        for bp_name, action_name in mapping.items():
            button, modifiers = button_press_parse(bp_name)
            if modifiers not in self._mapping:
                self._mapping[modifiers] = {}
            self._mapping[modifiers][button] = action_name
            self._modifiers.append((modifiers, button, action_name))

    def get_unique_action_for_modifiers(self, modifiers, button=1):
        """Gets a single, unique action name for a modifier mask.

        :param modifiers: a bitmask of GDK Modifier Constants
        :param button: the button number to require; defaults to 1.
        :rtype: string containing an action name, or None

        """
        try:
            modmap = self._mapping[modifiers]
            if len(modmap) > 1:
                return None
            return self._mapping[modifiers][button]
        except KeyError:
            return None

    def lookup(self, modifiers, button):
        """Look up a single pointer binding efficiently.

        :param modifiers: a bitmask of GDK Modifier Constants.
        :type modifiers: GdkModifierType or int
        :param button: a button number
        :type button: int
        :rtype: string containing an action name, or None

        """
        if modifiers not in self._mapping:
            return None
        return self._mapping[modifiers].get(button, None)

    def lookup_possibilities(self, modifiers):
        """Find potential actions, reachable via buttons or more modifiers

        :param modifiers: a bitmask of GDK Modifier Constants.
        :type modifiers: GdkModifierType or int
        :rtype: list

        Returns those actions which can be reached from the currently held
        modifier keys by either pressing a pointer button right now, or by
        holding down additional modifiers and then pressing a pointer button.
        If `modifiers` is empty, an empty list will be returned.

        Each element in the returned list is a 3-tuple of the form ``(MODS,
        BUTTON, ACTION NAME)``.

        """
        # This enables us to display:
        #  "<Ctrl>: with <Shift>+Button1, ACTION1; with Button3, ACTION2."
        # while the modifiers are pressed, but the button isn't. Also if
        # only a single possibility is returned, the handler should just
        # enter the mode as a springload (and display what just happened!)
        possibilities = []
        for possible, btn, action in self._modifiers:
            # Exclude possible bindings whose modifiers do not overlap
            if (modifiers & possible) != modifiers:
                continue
            # Include only exact matches, and those possibilities which can be
            # reached by pressing more modifier keys.
            if modifiers == possible or ~modifiers & possible:
                possibilities.append((possible, btn, action))
        return possibilities


class ButtonMappingEditor (Gtk.EventBox):
    """Editor for a prefs hash of pointer bindings mapped to action strings.

    """

    __gtype_name__ = 'ButtonMappingEditor'

    def __init__(self):
        """Initialise.
        """
        super(ButtonMappingEditor, self).__init__()
        import gui.application
        self.app = gui.application.get_app()
        self.actions = set()
        self.default_action = None
        self.bindings = None  #: dict of bindings being edited
        self.vbox = Gtk.VBox()
        self.add(self.vbox)

        # Display strings for action names
        self.action_labels = dict()

        # Model: combo cellrenderer's liststore
        ls = Gtk.ListStore(GObject.TYPE_STRING, GObject.TYPE_STRING)
        self.action_liststore = ls
        self.action_liststore_value_column = 0
        self.action_liststore_display_column = 1

        # Model: main list's liststore
        # This is reflected into self.bindings when it changes
        column_types = [GObject.TYPE_STRING] * 3
        ls = Gtk.ListStore(*column_types)
        self.action_column = 0
        self.bp_column = 1
        self.bpd_column = 2
        for sig in ("row-changed", "row-deleted", "row_inserted"):
            ls.connect(sig, self._liststore_updated_cb)
        self.liststore = ls

        # Bindings hash observers, external interface
        self.bindings_observers = []  #: List of cb(editor) callbacks

        # View: treeview
        scrolledwin = Gtk.ScrolledWindow()
        scrolledwin.set_shadow_type(Gtk.ShadowType.IN)
        tv = Gtk.TreeView()
        tv.set_model(ls)
        scrolledwin.add(tv)
        self.vbox.pack_start(scrolledwin, True, True, 0)
        tv.set_size_request(480, 320)
        tv.set_headers_clickable(True)
        self.treeview = tv
        self.selection = tv.get_selection()
        self.selection.connect("changed", self._selection_changed_cb)

        # Column 0: action name
        cell = Gtk.CellRendererCombo()
        cell.set_property("model", self.action_liststore)
        cell.set_property("text-column", self.action_liststore_display_column)
        cell.set_property("mode", Gtk.CellRendererMode.EDITABLE)
        cell.set_property("editable", True)
        cell.set_property("has-entry", False)
        cell.connect("changed", self._action_cell_changed_cb)
        # TRANSLATORS: Name of first column in the button map preferences.
        # TRANSLATORS: Refers to an action bound to a mod+button combination.
        col = Gtk.TreeViewColumn(_("Action"), cell)
        col.set_cell_data_func(cell, self._liststore_action_datafunc)
        col.set_min_width(150)
        col.set_resizable(False)
        col.set_expand(False)
        col.set_sort_column_id(self.action_column)
        tv.append_column(col)

        # Column 1: button press
        cell = Gtk.CellRendererText()
        cell.set_property("ellipsize", Pango.EllipsizeMode.END)
        cell.set_property("mode", Gtk.CellRendererMode.EDITABLE)
        cell.set_property("editable", True)
        cell.connect("edited", self._bp_cell_edited_cb)
        cell.connect("editing-started", self._bp_cell_editing_started_cb)
        # TRANSLATORS: Name of second column in the button map preferences.
        # TRANSLATORS: Column lists mod+button combinations (bound to actions)
        # TRANSLATORS: E.g. Button1 or Ctrl+Button2 or Alt+Button3
        col = Gtk.TreeViewColumn(_("Button press"), cell)
        col.add_attribute(cell, "text", self.bpd_column)
        col.set_expand(True)
        col.set_resizable(True)
        col.set_min_width(200)
        col.set_sort_column_id(self.bpd_column)
        tv.append_column(col)

        # List editor toolbar
        list_tools = Gtk.Toolbar()
        list_tools.set_style(Gtk.ToolbarStyle.ICONS)
        list_tools.set_icon_size(widgets.ICON_SIZE_LARGE)
        context = list_tools.get_style_context()
        context.add_class("inline-toolbar")
        self.vbox.pack_start(list_tools, False, False, 0)

        # Add binding
        btn = Gtk.ToolButton()
        btn.set_tooltip_text(_("Add a new binding"))
        btn.set_icon_name("mypaint-add-symbolic")
        btn.connect("clicked", self._add_button_clicked_cb)
        list_tools.add(btn)

        # Remove (inactive if list is empty)
        btn = Gtk.ToolButton()
        btn.set_icon_name("mypaint-remove-symbolic")
        btn.set_tooltip_text(_("Remove the current binding"))
        btn.connect("clicked", self._remove_button_clicked_cb)
        list_tools.add(btn)
        self.remove_button = btn

        self._updating_model = False

    def set_actions(self, actions):
        """Sets the internal list of possible actions.

        :param actions: List of all possible action strings. The 0th
          entry in the list is the default.
        :type actions: indexable sequence

        """
        self.default_action = actions[0]
        self.actions = set(actions)
        labels_list = sorted((self._get_action_label(a), a) for a in actions)
        self.action_liststore.clear()
        for label, act in labels_list:
            self.action_labels[act] = label
            self.action_liststore.append((act, label))

    def _liststore_action_datafunc(self, column, cell, model, iter,
                                   *user_data):
        action_name = model.get_value(iter, self.action_column)
        label = self.action_labels.get(action_name, action_name)
        cell.set_property("text", label)

    def _get_action_label(self, action_name):
        # Get a displayable (and translated) string for an action name
        handler_type, handler = get_handler_object(self.app, action_name)
        action_label = action_name
        if handler_type == 'gtk_action':
            action_label = handler.get_label()
        elif handler_type == 'popup_state':
            action_label = handler.label
        elif handler_type == 'mode_class':
            action_label = handler.get_name()
            if handler.ACTION_NAME is not None:
                action = self.app.find_action(handler.ACTION_NAME)
                if action is not None:
                    action_label = action.get_label()
        if action_label is None:
            action_label = ""  # Py3+: str cannot be compared to None
        return action_label

    def set_bindings(self, bindings):
        """Sets the mapping of binding names to actions.

        :param bindings: Mapping of pointer binding names to their actions. A
          reference is kept internally, and the entries will be
          modified.
        :type bindings: dict of bindings being edited

        The binding names in ``bindings`` will be canonicalized from the older
        ``<Control>`` prefix to ``<Primary>`` if supported by this Gtk.

        """
        tmp_bindings = dict(bindings)
        bindings.clear()
        for bp_name, action_name in tmp_bindings.items():
            bp_name = button_press_name(*button_press_parse(bp_name))
            bindings[bp_name] = action_name
        self.bindings = bindings
        self._bindings_changed_cb()

    def _bindings_changed_cb(self):
        """Updates the editor list to reflect the prefs hash changing.
        """
        self._updating_model = True
        self.liststore.clear()
        for bp_name, action_name in self.bindings.items():
            bp_displayname = button_press_displayname(
                *button_press_parse(bp_name))
            self.liststore.append((action_name, bp_name, bp_displayname))
        self._updating_model = False
        self._update_list_buttons()

    def _liststore_updated_cb(self, ls, *args, **kwargs):
        if self._updating_model:
            return
        iter = ls.get_iter_first()
        self.bindings.clear()
        while iter is not None:
            bp_name, action = ls.get(iter, self.bp_column, self.action_column)
            if action in self.actions and bp_name is not None:
                self.bindings[bp_name] = action
            iter = ls.iter_next(iter)
        self._update_list_buttons()
        for func in self.bindings_observers:
            func(self)

    def _selection_changed_cb(self, selection):
        if self._updating_model:
            return
        self._update_list_buttons()

    def _update_list_buttons(self):
        is_populated = len(self.bindings) > 0
        has_selected = self.selection.count_selected_rows() > 0
        self.remove_button.set_sensitive(is_populated and has_selected)

    def _add_button_clicked_cb(self, button):
        added_iter = self.liststore.append((self.default_action, None, None))
        self.selection.select_iter(added_iter)
        added_path = self.liststore.get_path(added_iter)
        focus_col = self.treeview.get_column(self.action_column)
        self.treeview.set_cursor_on_cell(added_path, focus_col, None, True)

    def _remove_button_clicked_cb(self, button):
        if self.selection.count_selected_rows() > 0:
            ls, selected = self.selection.get_selected()
            ls.remove(selected)

    ## "Controller" callbacks

    def _action_cell_changed_cb(self, combo, path_string, new_iter, *etc):
        action_name = self.action_liststore.get_value(
            new_iter,
            self.action_liststore_value_column
        )
        iter = self.liststore.get_iter(path_string)
        self.liststore.set_value(iter, self.action_column, action_name)
        self.treeview.columns_autosize()
        # If we don't have a button-press name yet, edit that next
        bp_name = self.liststore.get_value(iter, self.bp_column)
        if bp_name is None:
            focus_col = self.treeview.get_column(self.bp_column)
            tree_path = Gtk.TreePath(path_string)
            self.treeview.set_cursor_on_cell(tree_path, focus_col, None, True)

    def _bp_cell_edited_cb(self, cell, path, bp_name):
        iter = self.liststore.get_iter(path)
        bp_displayname = button_press_displayname(*button_press_parse(bp_name))
        self.liststore.set_value(iter, self.bp_column, bp_name)
        self.liststore.set_value(iter, self.bpd_column, bp_displayname)

    def _bp_cell_editing_started_cb(self, cell, editable, path):
        iter = self.liststore.get_iter(path)
        action_name = self.liststore.get_value(iter, self.action_column)
        bp_name = self.liststore.get_value(iter, self.bp_column)
        bp_displayname = button_press_displayname(*button_press_parse(bp_name))

        editable.set_sensitive(False)
        dialog = Gtk.Dialog()
        dialog.set_modal(True)
        dialog.set_title(_("Edit binding for '%s'") % action_name)
        dialog.set_transient_for(self.get_toplevel())
        dialog.set_position(Gtk.WindowPosition.CENTER_ON_PARENT)
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                           Gtk.STOCK_OK, Gtk.ResponseType.OK)
        dialog.set_default_response(Gtk.ResponseType.OK)
        dialog.connect("response", self._bp_edit_dialog_response_cb, editable)
        dialog.ok_btn = dialog.get_widget_for_response(Gtk.ResponseType.OK)
        dialog.ok_btn.set_sensitive(bp_name is not None)

        evbox = Gtk.EventBox()
        evbox.set_border_width(12)
        evbox.connect("button-press-event", self._bp_edit_box_button_press_cb,
                      dialog, editable)
        evbox.connect("enter-notify-event", self._bp_edit_box_enter_cb)

        table = Gtk.Table(3, 2)
        table.set_row_spacings(12)
        table.set_col_spacings(12)

        row = 0
        label = Gtk.Label()
        label.set_alignment(0, 0.5)
        # TRANSLATORS: Part of interface when adding a new button map binding.
        # TRANSLATORS: It's a label for the action part of the combination.
        # TRANSLATORS: Probably always the same as the column name
        # TRANSLATORS: "Action" with a trailing ":" or lang-specific symbol
        label.set_text(_("Action:"))
        table.attach(label, 0, 1, row, row + 1, Gtk.AttachOptions.FILL)

        label = Gtk.Label()
        label.set_alignment(0, 0.5)
        label.set_text(str(action_name))
        table.attach(
            label, 1, 2, row, row + 1,
            Gtk.AttachOptions.FILL | Gtk.AttachOptions.EXPAND)

        row += 1
        label = Gtk.Label()
        label.set_alignment(0, 0.5)
        # TRANSLATORS: Part of interface when adding a new button map binding.
        # TRANSLATORS: It's a label for the mod+button part of the combination.
        # TRANSLATORS: Probably always the same as "Button press" (column name)
        # TRANSLATORS: but with a trailing ":" or other lang-specific symbol.
        label.set_text(_("Button press:"))
        table.attach(label, 0, 1, row, row + 1, Gtk.AttachOptions.FILL)

        label = Gtk.Label()
        label.set_alignment(0, 0.5)
        label.set_text(str(bp_displayname))
        dialog.bp_name = bp_name
        dialog.bp_name_orig = bp_name
        dialog.bp_label = label
        table.attach(
            label, 1, 2, row, row + 1,
            Gtk.AttachOptions.FILL | Gtk.AttachOptions.EXPAND)

        row += 1
        label = Gtk.Label()
        label.set_size_request(300, 75)
        label.set_alignment(0, 0)
        label.set_line_wrap(True)
        dialog.hint_label = label
        self._bp_edit_dialog_set_standard_hint(dialog)
        table.attach(
            label, 0, 2, row, row + 1,
            Gtk.AttachOptions.FILL | Gtk.AttachOptions.EXPAND,
            Gtk.AttachOptions.FILL | Gtk.AttachOptions.EXPAND,
            0, 12)

        evbox.add(table)
        dialog.get_content_area().pack_start(evbox, True, True, 0)
        evbox.show_all()

        dialog.show()

    def _bp_edit_dialog_set_error(self, dialog, markup):
        dialog.hint_label.set_markup(
            "<span foreground='red'>%s</span>" % markup)

    def _bp_edit_dialog_set_standard_hint(self, dialog):
        markup = _("Hold down modifier keys, and press a button "
                   "over this text to set a new binding.")
        dialog.hint_label.set_markup(markup)

    def _bp_edit_box_enter_cb(self, evbox, event):
        window = evbox.get_window()
        disp = window.get_display()
        try:  # Wayland themes are a bit incomplete
            cursor = Gdk.Cursor.new_for_display(disp, Gdk.CursorType.CROSSHAIR)
            window.set_cursor(cursor)
        except Exception:
            logger.exception("Cursor setting failed")  # and otherwise ignore

    def _bp_edit_dialog_response_cb(self, dialog, response_id, editable):
        if response_id == Gtk.ResponseType.OK:
            if dialog.bp_name is not None:
                editable.set_text(dialog.bp_name)
            editable.editing_done()
        editable.remove_widget()
        dialog.destroy()

    def _bp_edit_box_button_press_cb(self, evbox, event, dialog, editable):
        modifiers = event.state & Gtk.accelerator_get_default_mod_mask()
        bp_name = button_press_name(event.button, modifiers)
        bp_displayname = button_press_displayname(event.button, modifiers)
        if modifiers == 0 and event.button == 1:
            self._bp_edit_dialog_set_error(
                dialog,
                # TRANSLATORS: "fixed" in the sense of "static" -
                # TRANSLATORS: something which cannot be changed
                _("{button} cannot be bound without modifier keys "
                  "(its meaning is fixed, sorry)")
                .format(
                    button=lib.xml.escape(bp_displayname),
                ),
            )
            dialog.ok_btn.set_sensitive(False)
            return
        action = None
        if bp_name != dialog.bp_name_orig:
            action = self.bindings.get(bp_name, None)
        if action is not None:
            action_label = self.action_labels.get(action, action)
            self._bp_edit_dialog_set_error(
                dialog,
                _("{button_combination} is already bound "
                  "to the action '{action_name}'")
                .format(
                    button_combination=lib.xml.escape(str(bp_displayname)),
                    action_name=lib.xml.escape(str(action_label)),
                ),
            )

            dialog.ok_btn.set_sensitive(False)
        else:
            self._bp_edit_dialog_set_standard_hint(dialog)
            dialog.bp_name = bp_name
            dialog.bp_label.set_text(str(bp_displayname))
            dialog.ok_btn.set_sensitive(True)
            dialog.ok_btn.grab_focus()
