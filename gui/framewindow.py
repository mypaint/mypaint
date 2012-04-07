
import gtk

from gettext import gettext as _

import windowing
from lib import tiledsurface # For the tile size

class Window(windowing.Dialog):
    def __init__(self, app):
        buttons = (gtk.STOCK_OK, gtk.RESPONSE_ACCEPT)
        windowing.Dialog.__init__(self, app, ("Frame"), app.drawWindow, buttons=buttons)

        self.callbacks_active = False

        x, y, w, h = self.app.doc.model.get_frame()

        self.width_adj  = gtk.Adjustment(w, upper=32000, lower=1, step_incr=1, page_incr=128)
        self.height_adj = gtk.Adjustment(h, upper=32000, lower=1, step_incr=1, page_incr=128)

        self.width_adj.connect('value-changed', self.on_size_adjustment_changed)
        self.height_adj.connect('value-changed', self.on_size_adjustment_changed)

        self.app.doc.model.frame_observers.append(self.on_frame_changed)

        self._init_ui()

    def _init_ui(self):
        height_label = gtk.Label(_('Height'))
        width_label = gtk.Label(_('Width'))

        height_entry = gtk.SpinButton(self.height_adj)
        width_entry = gtk.SpinButton(self.width_adj)

        size_table = gtk.Table(2, 2)
        size_table.attach(width_label, 0, 1, 0, 1)
        size_table.attach(height_label, 0, 1, 1, 2)
        size_table.attach(width_entry, 1, 2, 0, 1)
        size_table.attach(height_entry, 1, 2, 1, 2)

        crop_layer_button = gtk.Button(_('Crop to active layer bounds'))
        crop_document_button = gtk.Button(_('Crop to document bounds'))
        crop_layer_button.connect('clicked', self.crop_frame_cb, 'CropFrameToLayer')
        crop_document_button.connect('clicked', self.crop_frame_cb, 'CropFrameToDocument')

        hint_label = gtk.Label(_('Hold Ctrl-Shift-Space to move the frame.'))

        self.enable_button = gtk.CheckButton(_('Enabled'))
        self.enable_button.connect('toggled', self.on_frame_toggled)
        enabled = self.app.doc.model.frame_enabled
        self.enable_button.set_active(enabled)

        top_vbox = self.get_content_area()
        top_vbox.pack_start(size_table)
        top_vbox.pack_start(hint_label)
        top_vbox.pack_start(crop_layer_button)
        top_vbox.pack_start(crop_document_button)
        top_vbox.pack_start(self.enable_button)

        self.connect('response', self.on_response)

    def on_response(self, dialog, response_id):
        if response_id == gtk.RESPONSE_ACCEPT:
            self.hide()

    # FRAME
    def crop_frame_cb(self, button, command):
        if command == 'CropFrameToLayer':
            bbox = self.app.doc.model.get_current_layer().get_bbox()
        elif command == 'CropFrameToDocument':
            bbox = self.app.doc.model.get_bbox()
        else: assert 0
        self.app.doc.model.set_frame(*bbox)

    def on_frame_toggled(self, button):
        """Update the frame state in the model."""
        if self.callbacks_active:
            return

        self.app.doc.model.set_frame_enabled(button.get_active())

    def on_size_adjustment_changed(self, adjustment):
        """Update the frame size in the model."""
        if self.callbacks_active:
            return

        width = int(self.width_adj.get_value())
        height = int(self.height_adj.get_value())

        self.app.doc.model.set_frame(width=width, height=height)

    def on_frame_changed(self):
        """Update the UI to reflect the model."""
        self.callbacks_active = True # Prevent callback loops

        x, y, w, h = self.app.doc.model.get_frame()
        self.width_adj.set_value(w)
        self.height_adj.set_value(h)
        enabled = self.app.doc.model.frame_enabled
        self.enable_button.set_active(enabled)

        self.callbacks_active = False
