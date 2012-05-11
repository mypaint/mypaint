import math
import gobject
import gtk
from gtk import gdk


class DragFunc:
    """Base class for canvas pointer drag operations.
    """

    cursor = gdk.BOGOSITY
    # Currently just an active cursor
    # If these become modes, need to add a cursor for the inactive state too

    def __init__(self, doc, drawwindow=None, mode=None):
        self.mode = mode
        self.drawwindow = drawwindow
        self._doc = doc

    @property
    def doc(self):
        return self._doc

    @property
    def tdw(self):
        return self._doc.tdw

    @property
    def model(self):
        return self._doc.model

    def on_start(self, modifier):
        pass

    def on_update(self, dx, dy, x, y):
        pass

    def on_stop(self):
        pass


class PanViewDragFunc (DragFunc):

    cursor = gdk.FLEUR

    def on_update(self, dx, dy, x, y):
        self.tdw.scroll(-dx, -dy)
        # Accelerated scrolling: Highly comfortable to me, but lots of
        # negative feedback from users.  Certainly should be disabled
        # by default. Maybe add it back as preference one day.
        # https://gna.org/bugs/?16232
        #
        #self.tdw.scroll(-dx*3, -dy*3)


class RotateViewDragFunc (DragFunc):

    cursor = gdk.EXCHANGE

    def on_update(self, dx, dy, x, y):
        # calculate angular velocity from viewport center
        cx, cy = self.tdw.get_center()
        x, y = x-cx, y-cy
        phi2 = math.atan2(y, x)
        x, y = x-dx, y-dy
        phi1 = math.atan2(y, x)
        self.tdw.rotate(phi2-phi1)
        #self.tdw.rotate(2*math.pi*dx/300.0)


# Old rotozoom function, for historical interest:
#def dragfunc_rotozoom(self, dx, dy, x, y):
#    self.tdw.scroll(-dx, -dy)
#    self.tdw.zoom(math.exp(-dy/100.0))
#    self.tdw.rotate(2*math.pi*dx/500.0)



class ZoomViewDragFunc (DragFunc):

    cursor = gdk.SIZING

    def on_update(self, dx, dy, x, y):
        # workaround (should zoom at x=(first click point).x instead of cursor)
        self.tdw.scroll(-dx, -dy)
        self.tdw.zoom(math.exp(dy/100.0))


# Straight Lines, Ellipses and Curved Lines
class DynamicLineDragFunc (DragFunc):

    cursor = gdk.CROSS
    idle_srcid = None

    def on_start(self, modifier):
        self.lm = self.drawwindow.app.linemode
        self.lm.start_command(self.mode, modifier)

    def on_update(self, junk1, junk2, x, y):
        self.lm.update_position(x, y)
        if self.idle_srcid is None:
            self.idle_srcid = gobject.idle_add(self.idle_cb)

    def on_stop(self):
        self.idle_srcid = None
        self.lm.stop_command()

    def idle_cb(self):
        if self.idle_srcid is not None:
            self.idle_srcid = None
            self.lm.process_line()



class LayerMoveDragFunc (DragFunc):

    cursor = gdk.FLEUR

    model_x0 = None
    model_y0 = None
    final_model_dx = None
    final_model_dy = None
    idle_srcid = None

    def on_start(self, modifier):
        self.layer = self.model.get_current_layer()
        self.move = None

    def on_update(self, dx, dy, x, y):
        model_x, model_y = self.tdw.display_to_model(x, y)
        if self.move is None:
            self.move = self.layer.get_move(model_x, model_y)
            self.model_x0 = model_x
            self.model_y0 = model_y
        model_dx = model_x - self.model_x0
        model_dy = model_y - self.model_y0
        self.final_model_dx = model_dx
        self.final_model_dy = model_dy
        self.move.update(model_dx, model_dy)
        if self.idle_srcid is None:
            self.idle_srcid = gobject.idle_add(self.idle_cb)

    def on_stop(self):
        self.idle_srcid = None
        if self.move is None:
            return
        self.tdw.set_sensitive(False)
        self.tdw.set_override_cursor(gdk.Cursor(gdk.WATCH))
        while gtk.events_pending():
            gtk.main_iteration() # HACK to set the cursor
        # Finish up
        self.move.process(n=-1)
        self.move.cleanup()

        dx = self.final_model_dx
        dy = self.final_model_dy
        for stroke in self.layer.strokes:
            stroke.translate(dx, dy)

        self.offsets = None
        self.model.record_layer_move(self.layer, dx, dy)
        self.tdw.set_sensitive(True)
        self.tdw.set_override_cursor(None)

    def idle_cb(self):
        if self.idle_srcid is None:
            # Asked to terminate
            self.move.cleanup()
            return False
        if self.move.process():
            return True
        else:
            self.move.cleanup()
            self.idle_srcid = None
            return False


class MoveFrameDragFunc (DragFunc):

    cursor = gdk.ICON

    def on_update(self, dx, dy, x, y):
        if not self.model.frame_enabled:
            return

        x, y, w, h = self.model.get_frame()

        # Find the difference in document coordinates
        x0, y0 = self.tdw.display_to_model(x, y)
        x1, y1 = self.tdw.display_to_model(x+dx, y+dy)

        self.model.move_frame(dx=x1-x0, dy=y1-y0)


