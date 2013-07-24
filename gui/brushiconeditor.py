# -*- encoding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2009-2013 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import logging
logger = logging.getLogger(__name__)


from gettext import gettext as _

import gi
from gi.repository import Gtk
from gi.repository import Pango

import tileddrawwidget
import windowing
import lib.document
from document import CanvasController
from canvasevent import FreehandOnlyMode
import brushmanager
from lib.helpers import escape
from lib.observable import event


class BrushIconEditorWindow (windowing.SubWindow):
    """Main app subwindow for editing a brush's icon

    See `BrushIconEditor` for details of how this operates.
    """

    _TITLE_PREVIEWING = _('Brush Icon')
    _TITLE_EDITING = _('Brush Icon (editing)')

    def __init__(self):
        from application import get_app
        app = get_app()
        self._app = app
        windowing.SubWindow.__init__(self, app)
        self._editor = BrushIconEditor()
        self._editor.mode_changed += self._editor_mode_changed
        self.add(self._editor)
        self.set_title(self._TITLE_PREVIEWING)

    def _editor_mode_changed(self, editor, editing):
        if editing:
            self.set_title(self._TITLE_EDITING)
        else:
            self.set_title(self._TITLE_PREVIEWING)



class BrushIconEditor (Gtk.Grid):
    """Widget for previewing and editing a brush's icon at a large size

    The editor has two modes: previewing and editing.  In preview mode, the
    widget's view of the brush icon just tracks the current brush.  When the
    Edit button is clicked, the icon is locked for editing and made sensitive,
    and the user can switch brushes and colours as necessary to draw a pretty
    icon.  The Clear, Save and Revert buttons do what you'd expect; saving and
    reverting also exit the editing mode.

    The name of the brush which will be affected is shown at all times, along
    with an indication of the current mode.

    """

    ## Class constants

    _SCALE = 2
    _NO_BRUSH_NAME = _("No brush selected")
    _ICON_INVALID_TMPL = _(u'<b>%s</b>\n'
        '<small>Select a valid brush first</small>')
    _ICON_MODIFIED_TMPL = _(u'<b>%s</b> <i>(modified)</i>\n'
        u'<small>Changes are not yet saved</small>')
    _ICON_MODIFIABLE_TMPL = _(u'<b>%s</b> (editing)\n'
        u'<small>Paint with any brush or color</small>')
    _ICON_PREVIEWING_TMPL = _('<b>%s</b>\n'
        u'<small>Click ‘Edit’ to make changes to the icon</small>')


    ## Construction

    def __init__(self):
        Gtk.Grid.__init__(self)
        self.set_row_spacing(6)
        self.set_column_spacing(12)
        from application import get_app
        app = get_app()
        self._app = app
        self._bm = app.brushmanager
        self.set_border_width(12)
        self._bm.brush_selected += self._brush_selected_cb
        self._brush_to_edit = None
        self._preview_modified = False
        self._model = lib.document.Document(self._app.brush)
        self._model.canvas_observers.append(self._preview_modified_cb)
        self._init_widgets()


    @staticmethod
    def _make_image_button(text, stock_id, cb):
        b = Gtk.Button(text)
        i = Gtk.Image()
        i.set_from_stock(stock_id, Gtk.IconSize.BUTTON)
        b.set_image(i)
        b.set_image_position(Gtk.PositionType.TOP)
        b.connect("clicked", cb)
        b.set_can_focus(False)
        b.set_can_default(False)
        return b


    def _init_widgets(self):
        # Icon preview and edit TDW
        self._tdw = tileddrawwidget.TiledDrawWidget()
        self._tdw.set_model(self._model)
        self._tdw.set_size_request(brushmanager.PREVIEW_W*self._SCALE,
                                  brushmanager.PREVIEW_H*self._SCALE)
        self._tdw.scale = float(self._SCALE)
        self._tdw.scroll_on_allocate = False
        self._tdw.pixelize_threshold = 0
        tdw_align = Gtk.Alignment(xalign=0.5, yalign=0.0,
                                  xscale=0.0, yscale=0.0)
        tdw_align.add(self._tdw)
        self.attach(tdw_align, 0, 0, 1, 1)

        ctrlr = CanvasController(self._tdw)
        ctrlr.init_pointer_events()
        ctrlr.modes.default_mode_class = FreehandOnlyMode

        # Brush name label
        lbl = Gtk.Label()
        lbl.set_alignment(0.5, 0.0)
        lbl.set_justify(Gtk.Justification.CENTER)
        lbl_tmpl = self._ICON_PREVIEWING_TMPL
        lbl.set_markup(lbl_tmpl % (escape(self._NO_BRUSH_NAME),))
        self.attach(lbl, 0, 1, 1, 1)
        self.brush_name_label = lbl

        # Action buttons
        button_box = Gtk.VButtonBox()
        button_box.set_homogeneous(False)
        button_box.set_layout(Gtk.ButtonBoxStyle.START)

        # TRANSLATORS: begin editing a brush's preview icon
        b = self._make_image_button(_('Edit'), Gtk.STOCK_EDIT,
                                    self._edit_cb)
        b.set_tooltip_text(_("Begin editing this preview icon"))
        button_box.pack_start(b, expand=False)
        self._edit_button = b

        # TRANSLATORS: clear the brush preview icon being edited
        b = self._make_image_button(_('Clear'), Gtk.STOCK_CLEAR,
                                    self._clear_cb)
        b.set_tooltip_text(_("Clear the preview icon"))
        button_box.pack_start(b, expand=False)
        self._clear_button = b

        #lbl = Gtk.Label(_("Use any brush and color when editing"))
        #lbl.set_line_wrap(Pango.WrapMode.WORD_CHAR)
        #button_box.pack_start(lbl, expand=False)

        # TRANSLATORS: revert edits to a brush icon
        b = self._make_image_button(_('Revert'), Gtk.STOCK_REVERT_TO_SAVED,
                                    self._revert_cb)
        b.set_tooltip_text(_("Discard changes, and cancel editing"))
        button_box.pack_start(b, expand=False)
        button_box.set_child_secondary(b, True)
        self._revert_button = b

        # TRANSLATORS: save edits to a brush icon
        b = self._make_image_button(_('Save'), Gtk.STOCK_SAVE,
                                    self._save_cb)
        b.set_tooltip_text(_("Save this preview icon, and finish editing"))
        button_box.pack_start(b, expand=False)
        button_box.set_child_secondary(b, True)
        self._save_button = b

        self.attach(button_box, 1, 0, 1, 2)

        self.connect_after("show", self._show_cb)

        mb = self._bm.selected_brush
        preview = mb.preview
        self._set_preview_pixbuf(preview)
        name = mb.name
        if name is None:
            name = self._NO_BRUSH_NAME
        self.brush_name_label.set_markup(lbl_tmpl % (escape(name),))


    ## Public subscriber interface

    @event
    def mode_changed(self, editing):
        """Event: called when the mode changes

        :param editing: True if the editor is now in edit-mode.
        """


    ## Event handling

    def _show_cb(self, widget):
        self._update_widgets()


    def _preview_modified_cb(self, x, y, w, h):
        """Handles changes made to the preview canvas"""
        self._preview_modified = True
        self._update_widgets()


    def _brush_selected_cb(self, bm, managed_brush, brushinfo):
        """Updates the brush icon preview if it is not in edit mode"""
        if not self._brush_to_edit:
            self._set_preview_pixbuf(managed_brush.preview)
            self._update_widgets()


    ## Button callbacks

    def _clear_cb(self, button):
        assert self._brush_to_edit
        self._tdw.doc.clear_layer()


    def _edit_cb(self, button):
        mb = self._bm.selected_brush
        assert not self._brush_to_edit
        self._brush_to_edit = mb
        logger.debug("Started editing %r", self._brush_to_edit)
        self._update_widgets()
        self.mode_changed(True)


    def _revert_cb(self, button):
        assert self._brush_to_edit
        logger.debug("Reverted edits to %r", self._brush_to_edit)
        preview = self._bm.selected_brush.preview
        self._set_preview_pixbuf(preview)
        self._brush_to_edit = None
        self._update_widgets()
        self.mode_changed(False)


    def _save_cb(self, button):
        pixbuf = self._get_preview_pixbuf()
        assert self._brush_to_edit is not None
        b = self._brush_to_edit
        assert b.name is not None
        b.preview = pixbuf
        try:
            b.save()
        except IOError, err:
            logger.warning("Failed to save brush: %r (recoverable!)", err)
        else:
            for brushes in self._bm.groups.itervalues():
                if b in brushes:
                    self._bm.brushes_changed(brushes)
            logger.info("Saved %r", b)
            self._brush_to_edit = None
            self._update_widgets()
            self._bm.select_brush(b)
            self.mode_changed(False)
            return
        # Failed to save the icon.
        # This can happen if the user deletes a brush whose icon is being
        # edited. To recover, add the saved settings as a new brush
        logger.info("Failed to save preview, so saving cached settings"
                    "as a new brush")
        b = self._brush_to_edit.clone(name=None)
        group = brushmanager.NEW_BRUSH_GROUP
        brushes = self._bm.get_group_brushes(group)
        brushes.insert(0, b)
        b.save()
        b.persistent = True
        self._bm.brushes_changed(brushes)
        self._bm.select_brush(b)
        # Reveal the "New" group if needed
        ws = self._app.workspace
        ws.show_tool_widget("MyPaintBrushGroupTool", (group,))
        logger.info("Saved %r (full)", b)
        self._brush_to_edit = None
        self._update_widgets()
        self.mode_changed(False)


    ## Utility methods

    def _update_widgets(self):
        editing = self._brush_to_edit is not None
        if editing:
            brush = self._brush_to_edit
        else:
            brush = self._bm.selected_brush
        # Fairly rare: corresponds to no brush being selected on startup
        valid = brush.name is not None
        # Button states
        self._revert_button.set_sensitive(valid and editing)
        self._edit_button.set_sensitive(valid and not editing)
        self._clear_button.set_sensitive(valid and editing)
        self._save_button.set_sensitive(valid and editing)
        self._model.layer.locked = not (valid and editing)
        # Text to display in the various states
        if not valid:
            tmpl = self._ICON_INVALID_TMPL
        elif editing:
            if self._preview_modified:
                tmpl = self._ICON_MODIFIED_TMPL
            else:
                tmpl = self._ICON_MODIFIABLE_TMPL
        else:
            tmpl = self._ICON_PREVIEWING_TMPL
        # Update edit flag label
        name = brush.name
        if (not valid) or name is None:
            name = self._NO_BRUSH_NAME
        markup = tmpl % (escape(name),)
        self.brush_name_label.set_markup(markup)


    def _set_preview_pixbuf(self, pixbuf):
        if pixbuf is None:
            self._tdw.doc.clear()
        else:
            self._tdw.doc.load_from_pixbuf(pixbuf)
        self._preview_modified = False


    def _get_preview_pixbuf(self):
        w, h = brushmanager.PREVIEW_W, brushmanager.PREVIEW_H
        return self._tdw.doc.render_as_pixbuf(0, 0, w, h, alpha=False)


