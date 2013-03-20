
import os
USE_GTK3 = True
if os.environ.get('MYPAINT_ENABLE_GTK3', 1) in ("0", "NO", "no", "N", "n"):
    USE_GTK3 = False
print USE_GTK3

if USE_GTK3:
    import gi
    from gi.repository import Gdk, Gtk, GdkPixbuf

class GdkPixbufCompat(object):

    @staticmethod
    def save(pixbuf, path, type, **kwargs):

        if USE_GTK3:
            return pixbuf.savev(path, type, kwargs.keys(), kwargs.values())
        else:
            return pixbuf.save(path, type, kwargs)

    @staticmethod
    def new(colorspace, has_alpha, bps, width, height):

        if USE_GTK3:
            return GdkPixbuf.Pixbuf.new(colorspace, has_alpha, bps, width, height)
        else:
            return orig_gtk.gdk.Pixbuf(colorspace, has_alpha, bps, width, height)

class GdkCompat(object):

    @staticmethod
    def display_get_default():

        if USE_GTK3:
            display_manager = Gdk.DisplayManager.get()
            return display_manager.get_default_display()
        else:
            return orig_gtk.gdk.display_get_default()

    @staticmethod
    def keymap_get_default():
        if USE_GTK3:
            return Gdk.Keymap.get_default()
        else:
            return orig_gtk.gdk.keymap_get_default()

class GtkCompat(object):

    @staticmethod
    def recent_manager_get_default():
        if USE_GTK3:
            return Gtk.RecentManager.get_default()
        else:
            return orig_gtk.recent_manager_get_default()

    @staticmethod
    def settings_get_default():
        if USE_GTK3:
          return Gtk.Settings.get_default()
        else:
          return orig_gtk.settings_get_default()

    def accel_map_load(self, file):
        if USE_GTK3:
            return Gtk.AccelMap.load(file)
        else:
            return orig_gtk.accel_map_load(file)

    def accel_map_save(self, file):
        if USE_GTK3:
            return Gtk.AccelMap.save(file)
        else:
            return orig_gtk.accel_map_save(file)

    def accel_map_get(self):
        if USE_GTK3:
            return Gtk.AccelMap.get()
        else:
            return orig_gtk.accel_map_get()

    def accel_map_lookup_entry(self, accel_path):
        # Returns "a 2-tuple containing the keyval and modifier mask
        # corresponding to accel_path or None if not valid", like the GTK2
        # function.
        if USE_GTK3:
            found, accel_key = Gtk.AccelMap.lookup_entry(accel_path)
            if not found:
                return None
            keyval = accel_key.accel_key
            mods = accel_key.accel_mods
            return keyval, mods
        else:
            return orig_gtk.accel_map_lookup_entry(accel_path)

def get_gobject():
    if USE_GTK3:
        from gi.repository import GObject
        return GObject
    else:
        import gobject
        return gobject

def original_gtk():
    if USE_GTK3:
        print "Using GTK3"

        import gi
        import gi.pygtkcompat

        gi.pygtkcompat.enable()
        gi.pygtkcompat.enable_gtk(version='3.0')
    else:
        print "Using GTK2"

        import pygtk
        pygtk.require('2.0')

    import gtk
    return gtk

orig_gtk = original_gtk()
gdk = GdkCompat()
gdk.pixbuf = GdkPixbufCompat()
gtk = GtkCompat()
gobject = get_gobject()
