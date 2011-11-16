import math


class DragFunc:
    """Base class for canvas pointer drag operations.
    """

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

    def on_update(self, dx, dy, x, y):
        self.tdw.scroll(-dx, -dy)
        # Accelerated scrolling: Highly comfortable to me, but lots of
        # negative feedback from users.  Certainly should be disabled
        # by default. Maybe add it back as preference one day.
        # https://gna.org/bugs/?16232
        #
        #self.tdw.scroll(-dx*3, -dy*3)


class RotateViewDragFunc (DragFunc):

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

    def on_update(self, dx, dy, x, y):
        # workaround (should zoom at x=(first click point).x instead of cursor)
        self.tdw.scroll(-dx, -dy)
        self.tdw.zoom(math.exp(dy/100.0))


class LayerMoveDragFunc (DragFunc):

    def on_update(self, dx, dy, x, y):
        cr = self.tdw.get_model_coordinates_cairo_context()
        x0, y0 = cr.device_to_user(x, y)
        x1, y1 = cr.device_to_user(x+dx, y+dy)
        self.model.move_current_layer(dx=x1-x0, dy=y1-y0)


class MoveFrameDragFunc (DragFunc):

    def on_update(self, dx, dy, x, y):
        if not self.model.frame_enabled:
            return

        x, y, w, h = self.model.get_frame()

        # Find the difference in document coordinates
        cr = self.tdw.get_model_coordinates_cairo_context()
        x0, y0 = cr.device_to_user(x, y)
        x1, y1 = cr.device_to_user(x+dx, y+dy)

        self.model.move_frame(dx=x1-x0, dy=y1-y0)


