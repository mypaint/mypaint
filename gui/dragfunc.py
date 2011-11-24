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

    def __init__(self, doc):
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

    def on_start(self):
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



class LayerMoveDragFunc (DragFunc):

    cursor = gdk.CROSSHAIR

    model_x0 = None
    model_y0 = None
    final_model_dx = None
    final_model_dy = None
    idle_srcid = None

    def on_start(self):
        self.layer = self.model.get_current_layer()
        self.snapshot = None
        self.chunks = []
        self.chunks_i = -1
        self.offsets = None

    def on_update(self, dx, dy, x, y):
        cr = self.tdw.get_model_coordinates_cairo_context()
        model_x, model_y = cr.device_to_user(x, y)
        if self.snapshot is None:
            self.model_x0 = model_x
            self.model_y0 = model_y
            self.snapshot, self.chunks \
              = self.layer.begin_interactive_move(model_x, model_y)
            self.chunks_i = 0
        model_dx = model_x - self.model_x0
        model_dy = model_y - self.model_y0
        self.final_model_dx = model_dx
        self.final_final_dy = model_dy
        self.offsets = self.layer.update_interactive_move(model_dx, model_dy)
        self.chunks_i = 0
        if self.idle_srcid is None:
            self.idle_srcid = gobject.idle_add(self.idle_cb)

    def on_stop(self):
        self.idle_srcid = None
        if self.offsets is not None:
            if self.chunks_i < len(self.chunks):
                chunks = self.chunks[self.chunks_i:]
                self.chunks = []
                self.chunks_i = -1
                layer = self.layer
                self.tdw.set_sensitive(False)
                self.tdw.set_override_cursor(gdk.Cursor(gdk.WATCH))
                def process_remaining():
                    layer.process_interactive_move_queue(\
                        self.snapshot, chunks, self.offsets)
                    self.tdw.set_sensitive(True)
                    self.tdw.set_override_cursor(None)
                    self.offsets = None
                    return False
                gobject.idle_add(process_remaining)
            # TODO: make this undoable

    def idle_cb(self, k=200):
        if self.idle_srcid is None:
            # Asked to terminate
            return False
        assert self.chunks_i >= 0
        assert self.offsets is not None
        i = self.chunks_i
        if self.chunks_i >= len(self.chunks):
            self.idle_srcid = None
            return False
        chunks = self.chunks[i:i+k]
        self.layer.process_interactive_move_queue(
                self.snapshot, chunks, self.offsets)
        self.chunks_i += k
        if self.chunks_i < len(self.chunks):
            return True
        else:
            # Stop, but mark as restartable
            self.idle_srcid = None
            return False

class MoveFrameDragFunc (DragFunc):

    cursor = gdk.ICON

    def on_update(self, dx, dy, x, y):
        if not self.model.frame_enabled:
            return

        x, y, w, h = self.model.get_frame()

        # Find the difference in document coordinates
        cr = self.tdw.get_model_coordinates_cairo_context()
        x0, y0 = cr.device_to_user(x, y)
        x1, y1 = cr.device_to_user(x+dx, y+dy)

        self.model.move_frame(dx=x1-x0, dy=y1-y0)


