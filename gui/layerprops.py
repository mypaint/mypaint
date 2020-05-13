# This file is part of MyPaint.
# Copyright (C) 2017 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Properties UI showing the current layer."""


# Imports:

from __future__ import division, print_function

import logging
from collections import namedtuple

from lib.modes import STACK_MODES
from lib.modes import STANDARD_MODES
from lib.modes import PASS_THROUGH_MODE
from lib.modes import MODE_STRINGS
import lib.xml
from lib.gettext import C_
import gui.mvp

import cairo
from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import GdkPixbuf


# Module constants:

logger = logging.getLogger(__name__)


# Class defs:


_LayerFlagUIInfo = namedtuple("_LayerFlagUIInfo", [
    # View objects
    "togglebutton",
    "image",
    # Model details
    "property",
    # Mapping: 2-tuples, indexed by int(property)
    "togglebutton_active",
    "image_icon_name",
])


class LayerPropertiesUI (gui.mvp.BuiltUIPresenter, object):
    """Presents a widget for editing the current layer's properties.

    Implemented as a Pythonic MVP Presenter that observes the main
    document Model via its exposed lib.observable events.

    The View part is an opaque GTK widget that can be plugged into the
    rest of the UI anywhere.  It's instantiated on demand: its
    corresponding UI XML can be found in layerprops.glade in the same
    directory as this one.

    """

    # Class setting vars:

    _LAYER_MODE_TOOLTIP_MARKUP_TEMPLATE = "<b>{name}</b>\n{description}"
    _PREVIEW_SIZE = 256
    _BOOL_PROPERTIES = [
        _LayerFlagUIInfo(
            togglebutton="layer-locked-togglebutton",
            image="layer-locked-image",
            property="locked",
            togglebutton_active=[False, True],
            image_icon_name=[
                "mypaint-object-unlocked-symbolic",
                "mypaint-object-locked-symbolic",
            ],
        ),
        _LayerFlagUIInfo(
            togglebutton="layer-hidden-togglebutton",
            image="layer-hidden-image",
            property="visible",
            togglebutton_active=[True, False],
            image_icon_name=[
                "mypaint-object-hidden-symbolic",
                "mypaint-object-visible-symbolic",
            ],
        ),
    ]
    _FLAG_ICON_SIZE = Gtk.IconSize.LARGE_TOOLBAR

    # Initialization:

    def __init__(self, docmodel):
        object.__init__(self)
        self._docmodel = docmodel
        root = docmodel.layer_stack
        root.current_path_updated += self._m_layer_changed_cb
        root.layer_properties_changed += self._m_layer_props_changed_cb
        root.layer_thumbnail_updated += self._m_layer_thumbnail_updated_cb
        self._store = None

    def init_view(self):
        """Set initial state of the view objects."""

        # 3-column mode liststore (id, name, sensitive)
        store = Gtk.ListStore(int, str, bool)
        modes = STACK_MODES + STANDARD_MODES
        for mode in modes:
            label, desc = MODE_STRINGS.get(mode)
            store.append([mode, label, True])
        self._store = store
        self.view.layer_mode_combo.set_model(store)

        # The eye button is greyed out while the view is locked.
        lvm = self._docmodel.layer_view_manager
        lvm.current_view_changed += self._m_current_view_changed_cb

        # Update to the current state of the model
        self._m2v_all()

    # Accessors:

    @property
    def widget(self):
        """Get the main GTK widget of the view."""
        return self.view.layer_properties_widget

    @property
    def _layer(self):
        root = self._docmodel.layer_stack
        return root.current

    # Model monitoring and response:

    @gui.mvp.view_updater
    def _m_layer_changed_cb(self, root, layerpath):
        """Handle a change of the currently active layer."""
        self._set_name_entry_warning_flag(False)
        self._m2v_all()

    @gui.mvp.view_updater
    def _m_layer_props_changed_cb(self, root, layerpath, layer, changed):
        """Handle a change of layer properties."""
        if layer is not self._layer:
            return
        if "mode" in changed:
            self._m2v_mode()
        if "opacity" in changed:
            self._m2v_opacity()
        if "locked" in changed:
            info = [i for i in self._BOOL_PROPERTIES
                    if (i.property == "locked")][0]
            self._m2v_layer_flag(info)
        if "visible" in changed:
            info = [i for i in self._BOOL_PROPERTIES
                    if (i.property == "visible")][0]
            self._m2v_layer_flag(info)
        if "name" in changed:
            self._m2v_name()

    @gui.mvp.view_updater
    def _m_layer_thumbnail_updated_cb(self, root, layerpath, layer):
        """Handle the thumbnail of a layer changing."""
        if layer is not self._layer:
            return
        self._m2v_preview()

    @gui.mvp.view_updater
    def _m_current_view_changed_cb(self, lvm):
        self._m2v_layerview_locked()

    def _m2v_all(self):
        self._m2v_preview()
        self._m2v_name()
        self._m2v_mode()
        self._m2v_opacity()
        for info in self._BOOL_PROPERTIES:
            self._m2v_layer_flag(info)
        self._m2v_layerview_locked()

    def _m2v_preview(self):
        layer = self._layer
        if not layer:
            return
        preview = make_preview(layer.thumbnail, self._PREVIEW_SIZE)
        image = self.view.layer_preview_image
        image.set_from_pixbuf(preview)

    def _m2v_name(self):
        entry = self.view.layer_name_entry
        layer = self._layer

        if not layer:
            entry.set_sensitive(False)
            return
        elif not entry.get_sensitive():
            entry.set_sensitive(True)

        name = layer.name
        if name is None:
            name = layer.DEFAULT_NAME

        root = self._docmodel.layer_stack
        if not root.layer_properties_changed.calling_observers:
            entry.set_text(name)

    def _m2v_mode(self):
        combo = self.view.layer_mode_combo
        layer = self._layer

        if not layer:
            combo.set_sensitive(False)
            return
        elif not combo.get_sensitive():
            combo.set_sensitive(True)

        active_iter = None
        for row in self._store:
            mode = row[0]
            if mode == layer.mode:
                active_iter = row.iter
            row[2] = (mode in layer.PERMITTED_MODES)

        combo.set_active_iter(active_iter)

    def _m2v_opacity(self):
        adj = self.view.layer_opacity_adjustment
        scale = self.view.layer_opacity_scale
        layer = self._layer

        opacity_is_adjustable = not (
            layer is None
            or layer is self._docmodel.layer_stack
            or layer.mode == PASS_THROUGH_MODE
        )
        scale.set_sensitive(opacity_is_adjustable)
        if not opacity_is_adjustable:
            return

        percentage = layer.opacity * 100
        adj.set_value(percentage)

    def _m2v_layer_flag(self, info):
        layer = self._layer
        propval = getattr(layer, info.property)
        propval_idx = int(propval)

        togbut = getattr(self.view, info.togglebutton)
        new_active = bool(info.togglebutton_active[propval_idx])
        togbut.set_active(new_active)

        image = getattr(self.view, info.image)
        new_icon = str(info.image_icon_name[propval_idx])
        image.set_from_icon_name(new_icon, self._FLAG_ICON_SIZE)

    def _m2v_layerview_locked(self):
        lvm = self._docmodel.layer_view_manager
        sensitive = not lvm.current_view_locked
        btn = self.view.layer_hidden_togglebutton
        btn.set_sensitive(sensitive)

    # View monitoring and response (callback names defined in .glade XML):

    def _v_layer_mode_combo_query_tooltip_cb(self, combo, x, y, kbd, tooltip):

        label, desc = MODE_STRINGS.get(self._layer.mode, (None, None))
        if not (label and desc):
            return False
        template = self._LAYER_MODE_TOOLTIP_MARKUP_TEMPLATE
        markup = template.format(
            name = lib.xml.escape(label),
            description = lib.xml.escape(desc),
        )
        tooltip.set_markup(markup)
        return True

    @gui.mvp.model_updater
    def _v_layer_name_entry_changed_cb(self, entry):
        if not self._layer:
            return
        proposed_name = entry.get_text().strip()
        old_name = self._layer.name
        if proposed_name == old_name:
            self._set_name_entry_warning_flag(False)
            return

        self._docmodel.rename_current_layer(proposed_name)
        approved_name = self._layer.name
        self._set_name_entry_warning_flag(proposed_name != approved_name)

    @gui.mvp.model_updater
    def _v_layer_mode_combo_changed_cb(self, combo):
        if not self._layer:
            return
        old_mode = self._layer.mode
        store = combo.get_model()
        it = combo.get_active_iter()
        if it is None:
            return
        new_mode = store.get_value(it, 0)
        if new_mode == old_mode:
            return
        self._docmodel.set_current_layer_mode(new_mode)

    @gui.mvp.model_updater
    def _v_layer_opacity_adjustment_value_changed_cb(self, adjustment, *etc):
        if not self._layer:
            return
        opacity = adjustment.get_value() / 100.0
        self._docmodel.set_current_layer_opacity(opacity)

    @gui.mvp.model_updater
    def _v_layer_hidden_togglebutton_toggled_cb(self, btn):
        info = [i for i in self._BOOL_PROPERTIES
                if (i.property == "visible")][0]
        self._v2m_layer_flag(info)

    @gui.mvp.model_updater
    def _v_layer_locked_togglebutton_toggled_cb(self, btn):
        info = [i for i in self._BOOL_PROPERTIES
                if (i.property == "locked")][0]
        self._v2m_layer_flag(info)

    def _v2m_layer_flag(self, info):
        layer = self._layer
        if not layer:
            return
        togbut = getattr(self.view, info.togglebutton)
        togbut_active = bool(togbut.get_active())
        new_propval = bool(info.togglebutton_active.index(togbut_active))
        if bool(getattr(layer, info.property)) != new_propval:
            setattr(layer, info.property, new_propval)

        new_propval_idx = int(new_propval)
        image = getattr(self.view, info.image)
        new_icon = str(info.image_icon_name[new_propval_idx])
        image.set_from_icon_name(new_icon, self._FLAG_ICON_SIZE)

    # Utility methods:

    def _set_name_entry_warning_flag(self, show_warning):
        entry = self.view.layer_name_entry
        pos = Gtk.EntryIconPosition.SECONDARY
        warning_showing = entry.get_icon_name(pos)
        if show_warning:
            if not warning_showing:
                entry.set_icon_from_icon_name(pos, "dialog-warning")
                text = entry.get_text()
                if text.strip() == u"":
                    msg = C_(
                        "layer properties dialog: name entry: icon tooltip",
                        u"Layer names cannot be empty.",
                    )
                else:
                    msg = C_(
                        "layer properties dialog: name entry: icon tooltip",
                        u"Layer name is not unique.",
                    )
                entry.set_icon_tooltip_text(pos, msg)
        elif warning_showing:
            entry.set_icon_from_icon_name(pos, None)
            entry.set_icon_tooltip_text(pos, None)


class LayerPropertiesDialog (Gtk.Dialog):
    """Interim dialog for editing the current layer's properties."""
    # Expect this to be replaced with a popover hanging off the layers
    # dockable when the main UI workspace allows for that (floating
    # windows as GtkOverlay overlay children needed 1st)

    TITLE_TEXT = C_(
        "layer properties dialog: title",
        u"Layer Properties",
    )
    DONE_BUTTON_TEXT = C_(
        "layer properties dialog: done button",
        u"Done",
    )

    def __init__(self, parent, docmodel):
        flags = (
            Gtk.DialogFlags.MODAL |
            Gtk.DialogFlags.DESTROY_WITH_PARENT
        )
        Gtk.Dialog.__init__(
            self, self.TITLE_TEXT, parent, flags,
            (self.DONE_BUTTON_TEXT, Gtk.ResponseType.OK),
        )
        self.set_position(Gtk.WindowPosition.CENTER_ON_PARENT)
        self._ui = LayerPropertiesUI(docmodel)
        self.vbox.pack_start(self._ui.widget, True, True, 0)
        self.set_default_response(Gtk.ResponseType.OK)


# Helpers:

def make_preview(thumb, preview_size):
    """Convert a layer's thumbnail into a nice preview image."""

    # Check size
    check_size = 2
    while check_size < (preview_size / 6) and check_size < 16:
        check_size *= 2

    blank = GdkPixbuf.Pixbuf.new(
        GdkPixbuf.Colorspace.RGB, True, 8,
        preview_size, preview_size,
    )
    blank.fill(0x00000000)

    if thumb is None:
        thumb = blank

    # Make a square of chex
    preview = blank.composite_color_simple(
        dest_width=preview_size,
        dest_height=preview_size,
        interp_type=GdkPixbuf.InterpType.NEAREST,
        overall_alpha=255,
        check_size=check_size,
        color1=0xff707070,
        color2=0xff808080,
    )

    w = thumb.get_width()
    h = thumb.get_height()
    scale = preview_size / max(w, h)
    w *= scale
    h *= scale
    x = (preview_size - w) // 2
    y = (preview_size - h) // 2

    thumb.composite(
        dest=preview,
        dest_x=x,
        dest_y=y,
        dest_width=w,
        dest_height=h,
        offset_x=x,
        offset_y=y,
        scale_x=scale,
        scale_y=scale,
        interp_type=GdkPixbuf.InterpType.BILINEAR,
        overall_alpha=255,
    )

    # Add some very minor decorations..
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, preview_size, preview_size)
    cr = cairo.Context(surf)
    Gdk.cairo_set_source_pixbuf(cr, preview, 0, 0)
    cr.paint()

    cr.set_source_rgba(1, 1, 1, 0.1)
    cr.rectangle(0.5, 0.5, preview_size-1, preview_size-1)
    cr.set_line_width(1.0)
    cr.stroke()

    surf.flush()

    preview = Gdk.pixbuf_get_from_surface(
        surf,
        0, 0, preview_size, preview_size,
    )

    return preview
