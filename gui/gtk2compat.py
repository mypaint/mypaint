"""PyGTK compatibility layer.

To be removed as we migrate to Python-GObject's normal syntax. Do not write new
code in the PyGTK style now, and feel free to simplify "if pygtkcompat.USE_GTK3"
clauses in code elsewhere.

"""

import gi
from gi.repository import GObject
from gi.repository import Gdk
from gi.repository import Gtk
from gi.repository import GdkPixbuf

USE_GTK3 = True

class GdkPixbufCompat(object):

    @staticmethod
    def save(pixbuf, path, type, **kwargs):
        return pixbuf.savev(path, type, kwargs.keys(), kwargs.values())

    @staticmethod
    def new(colorspace, has_alpha, bps, width, height):
        return GdkPixbuf.Pixbuf.new(colorspace, has_alpha, bps, width, height)


class GdkCompat(object):

    @staticmethod
    def display_get_default():
        display_manager = Gdk.DisplayManager.get()
        return display_manager.get_default_display()

    @staticmethod
    def keymap_get_default():
        return Gdk.Keymap.get_default()


class GtkCompat(object):

    @staticmethod
    def recent_manager_get_default():
        return Gtk.RecentManager.get_default()

    @staticmethod
    def settings_get_default():
        return Gtk.Settings.get_default()

    def accel_map_load(self, file):
        return Gtk.AccelMap.load(file)

    def accel_map_save(self, file):
        return Gtk.AccelMap.save(file)

    def accel_map_get(self):
        return Gtk.AccelMap.get()

    def accel_map_lookup_entry(self, accel_path):
        # Returns "a 2-tuple containing the keyval and modifier mask
        # corresponding to accel_path or None if not valid", like the GTK2
        # function.
        found, accel_key = Gtk.AccelMap.lookup_entry(accel_path)
        if not found:
            return None
        keyval = accel_key.accel_key
        mods = accel_key.accel_mods
        return keyval, mods


def get_gobject():
    return GObject


def original_gtk():
    print "Using GTK3"
    import gi
    import gi.pygtkcompat
    gi.pygtkcompat.enable()
    gi.pygtkcompat.enable_gtk(version='3.0')
    import gtk
    return gtk

orig_gtk = original_gtk()
gdk = GdkCompat()
gdk.pixbuf = GdkPixbufCompat()
gtk = GtkCompat()
gobject = get_gobject()
