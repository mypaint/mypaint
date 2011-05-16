import gtk
gdk = gtk.gdk
from math import pi, sin, cos, sqrt, atan2, ceil
import struct
import cairo
import windowing
from layout import ElasticExpander
from lib.helpers import rgb_to_hsv, hsv_to_rgb, clamp
from gettext import gettext as _

CSTEP=0.007
PADDING=4

class GColorSelector(gtk.DrawingArea):
    def __init__(self, app):
        gtk.DrawingArea.__init__(self)
        self.color = (1,0,0)
        self.hsv = (0,1,1)
        self.app = app
        self.connect('expose-event', self.draw)
        self.set_events(gdk.BUTTON_PRESS_MASK |
                        gdk.BUTTON_RELEASE_MASK |
                        gdk.POINTER_MOTION_MASK |
                        gdk.ENTER_NOTIFY |
                        gdk.LEAVE_NOTIFY |
                        gdk.DROP_FINISHED |
                        gdk.DROP_START |
                        gdk.DRAG_STATUS |
                        gdk.PROXIMITY_IN_MASK |
                        gdk.PROXIMITY_OUT_MASK)
        # When the colour is chosen set it for the input device that was used
        # for the input events
        self.set_extension_events(gdk.EXTENSION_EVENTS_ALL)
        self.connect('button-press-event', self.on_button_press)
        self.connect('button-release-event', self.on_button_release)
        self.connect('configure-event', self.on_configure)
        self.connect('motion-notify-event', self.motion)
        self.device_pressed = None
        self.grabbed = False
        self.set_size_request(110, 100)
        self.has_tooltip_areas = False

    def test_move(self, x, y):
        """
        Return true during motion if the selection indicator
        should move to where the cursor is pointing.
        """
        return True

    def test_button_release(self, x, y):
        """
        Return true after a button-release-event if the colour under the
        pointer should be selected.
        """
        return True

    def move(self, x, y):
        pass

    def redraw_on_select(self):
        self.queue_draw()

    def motion(self,w, event):
        if self.has_tooltip_areas:
            self.update_tooltip(event.x, event.y)
        if not self.device_pressed:
            return
        if self.test_move(event.x, event.y):
            if not self.grabbed:
                self.grabbed = True
                self.grab_add()
            self.move(event.x,event.y)

    def on_configure(self,w, size):
        self.x_rel = size.x
        self.y_rel = size.y
        self.w = size.width
        self.h = size.height
        self.configure_calc()

    def configure_calc(self):
        pass

    def on_select(self,color):
        pass

    def update_tooltip(self, x, y):
        """Updates the tooltip during motion, if tooltips are zoned."""
        pass

    def get_color_at(self, x,y):
        return self.hsv, True

    def select_color_at(self, x,y):
        color, is_hsv = self.get_color_at(x,y)
        if color is None:
            return
        if is_hsv:
            self.hsv = color
            self.color = hsv_to_rgb(*color)
        else:
            self.color = color
            self.hsv = rgb_to_hsv(*color)
        if self.device_pressed:
            selected_color = self.color
            # Potential device change, therefore brush & colour change...
            self.app.doc.tdw.device_used(self.device_pressed)
            self.color = selected_color #... but *this* is what the user wants
        self.redraw_on_select()
        self.on_select(self.hsv)

    def set_color(self, hsv):
        self.hsv = hsv
        self.color = hsv_to_rgb(*hsv)
        self.queue_draw()

    def get_color(self):
        return self.color

    def on_button_press(self, w, event):
        self.press_x = event.x
        self.press_y = event.y
        # Remember the device that was clicked, tapped to the stylus, etc.  We
        # process any potential device changing on the button release to avoid
        # redrawing with colours associated with the newly picked bush during
        # the select.
        self.device_pressed = event.device

    def on_button_release(self,w, event):
        if self.device_pressed and self.test_button_release(event.x, event.y):
            self.select_color_at(event.x, event.y)
        self.device_pressed = None
        if self.grabbed:
            self.grab_remove()
            self.grabbed = False

    def draw(self,w,event):
        if not self.window:
            return
        cr = self.window.cairo_create()
        cr.set_source_rgb(*self.color)
        cr.rectangle(PADDING,PADDING, self.w-2*PADDING, self.h-2*PADDING)
        cr.fill()

class RectSlot(GColorSelector):
    def __init__(self,app,color=(1.0,1.0,1.0),size=32):
        GColorSelector.__init__(self, app)
        self.color = color
        self.set_size_request(size,size)

class RecentColors(gtk.HBox):
    def __init__(self, app):
        gtk.HBox.__init__(self)
        self.set_border_width(4)
        self.N = N = app.ch.num_colors
        self.slots = []
        for i in xrange(N):
            slot = RectSlot(app)
            slot.on_select = self.slot_selected
            self.pack_start(slot, expand=True)
            self.slots.append(slot)
        self.app = app
        app.ch.color_pushed_observers.append(self.refill_slots)
        self.set_tooltip_text(_("Recently used colors"))
        self.refill_slots(None)
        self.show_all()

    def slot_selected(self,color):
        self.on_select(color)

    def on_select(self,color):
        pass

    def refill_slots(self, *junk):
        for hsv,slot in zip(self.app.ch.colors, reversed(self.slots)):
            slot.set_color(hsv)

    def set_color(self, hsv):
        pass

CIRCLE_N = 12.0
SLOTS_N = 5
NEUTRAL_MID_GREY  = (0.5, 0.5, 0.5)  #: Background neutral mid grey, intended to permit colour comparisons without theme colours distracting and imposing themselves
NEUTRAL_DARK_GREY = (0.46, 0.46, 0.46) #: Slightly distinct outlinefor colour pots, intended to reduce "1+1=3 (or more)" effects

AREA_SQUARE = 1  #: Central saturation/value square
AREA_INSIDE = 2  #: Grey area inside the rings
AREA_SAMPLE = 3  #: Inner ring of color sampler boxes (hue)
AREA_CIRCLE = 4  #: Outer gradiated ring (hue)
AREA_COMPARE = 5  #: Current/previous comparison semicircles
AREA_OUTSIDE = 6  #: Blank outer area, outer space

CIRCLE_TOOLTIPS = \
  { AREA_SQUARE: _('Change Saturation and Value'),
    AREA_INSIDE: None,
    AREA_SAMPLE: _('Harmony color switcher'),
    AREA_CIRCLE: _('Change Hue'),
    AREA_COMPARE: _('Current color vs. Last Used'),
    AREA_OUTSIDE: None,
    }

sq32 = sqrt(3)/2
sq33 = sqrt(3)/3
sq36 = sq33/2
sq22 = sqrt(2)/2

def try_put(list, item):
    if not item in list:
        list.append(item)

class CircleSelector(GColorSelector):
    def __init__(self,app,color=(1,0,0)):
        GColorSelector.__init__(self,app)
        self.color = color
        self.hsv = rgb_to_hsv(*color)

        self.samples = []      # [(h,s,v)] -- list of `harmonic' colors
        self.last_line = None  # 
        app.ch.color_pushed_observers.append(self.color_pushed_cb)

        self.has_tooltip_areas = True
        self.previous_tooltip_area = None
        self.previous_tooltip_xy = None

    def color_pushed_cb(self, pushed_color):
        self.queue_draw()

    def has_harmonies_visible(self):
        return self.app.preferences.get("colorsampler.complementary", False) \
            or self.app.preferences.get("colorsampler.triadic", False) \
            or self.app.preferences.get("colorsampler.double_comp", False) \
            or self.app.preferences.get("colorsampler.split_comp", False) \
            or self.app.preferences.get("colorsampler.analogous", False) \
            or self.app.preferences.get("colorsampler.square", False)

    def get_previous_color(self):
        return self.color

    def update_tooltip(self, x, y):
        area = self.area_at(x, y)
        if self.previous_tooltip_area is None:
            tooltip = CIRCLE_TOOLTIPS.get(area, None)
            self.set_tooltip_text(tooltip)
            self.previous_tooltip_area = area, x, y
        else:
            old_area, old_x, old_y = self.previous_tooltip_area
            if area != old_area or abs(old_x - x) > 20 or abs(old_y - y) > 20:
                self.set_tooltip_text(None)
                self.previous_tooltip_area = None

    def calc_line(self, angle):
        x1 = self.x0 + self.r2*cos(-angle)
        y1 = self.y0 + self.r2*sin(-angle)
        x2 = self.x0 + self.r3*cos(-angle)
        y2 = self.y0 + self.r3*sin(-angle)
        return x1,y1,x2,y2

    def configure_calc(self):
        padding = 5
        self.x0 = self.x_rel + self.w/2.0
        self.y0 = self.y_rel + self.h/2.0
        self.M = M = min(self.w,self.h) - (2*padding)
        self.r3 = 0.5*M     # outer radius of hue ring
        if not self.has_harmonies_visible():
            self.r2 = 0.42*M  # inner radius of hue ring
            self.rd = 0.42*M  # and size of square
        else:
            self.r2 = 0.44*M  # line between hue ring & harmony samplers
            self.rd = 0.34*M  # size of square & inner edge of harmony samplers
        self.m = self.rd/sqrt(2.0)
        self.circle_img = None
        self.stroke_width = 0.01*M

    def set_color(self, hsv, redraw=True):
        self.hsv = hsv
        self.color = hsv_to_rgb(*hsv)
        if redraw:
            self.queue_draw()

    def test_move(self, x, y):
        from_area = self.area_at(self.press_x,self.press_y)
        to_area = self.area_at(x, y)
        return from_area in (AREA_CIRCLE, AREA_SQUARE)

    def test_button_release(self, x, y):
        from_area = self.area_at(self.press_x,self.press_y)
        to_area = self.area_at(x, y)
        return from_area == to_area \
            and from_area in [AREA_SAMPLE, AREA_CIRCLE, AREA_SQUARE]

    def nearest_move_target(self, x,y):
        """
        Returns the nearest (x, y) the selection indicator can move to during a
        move with the button held down.
        """
        area_source = self.area_at(self.press_x, self.press_y)
        area = self.area_at(x,y)
        if area == area_source and area in [AREA_CIRCLE, AREA_SQUARE]:
            return x,y
        dx = x-self.x0
        dy = y-self.y0
        d = sqrt(dx*dx+dy*dy)
        x1 = dx/d
        y1 = dy/d
        if area_source == AREA_CIRCLE:
            rx = self.x0 + (self.r2+3.0)*x1
            ry = self.y0 + (self.r2+3.0)*y1
        else:
            m = self.m
            dx = clamp(dx, -m, m)
            dy = clamp(dy, -m, m)
            rx = self.x0 + dx
            ry = self.y0 + dy
        return rx,ry

    def move(self,x,y):
        x_,y_ = self.nearest_move_target(x,y)
        self.select_color_at(x_,y_)

    def area_at(self, x,y):
        dx = x-self.x0
        dy = y-self.y0
        d = sqrt(dx*dx+dy*dy)
        sq2 = sqrt(2.0)
        cmp_M = self.M/2.0
        cmp_r = (sq2-1)*cmp_M/(2*sq2)
        cmp_x0 = self.x0+cmp_M-cmp_r
        cmp_y0 = self.y0+cmp_M-cmp_r
        cmp_dx = x - cmp_x0
        cmp_dy = y - cmp_y0
        if cmp_r > sqrt(cmp_dx*cmp_dx + cmp_dy*cmp_dy):
            return AREA_COMPARE
        elif d > self.r3:
            return AREA_OUTSIDE
        elif self.r2 < d <= self.r3:
            return AREA_CIRCLE
        elif abs(dx)<ceil(self.m) and abs(dy)<ceil(self.m):
            return AREA_SQUARE
        elif self.rd < d <= self.r2:
            return AREA_SAMPLE
        else:
            return AREA_INSIDE

    def get_color_at(self, x,y):
        area = self.area_at(x,y)
        if area == AREA_CIRCLE:
            h,s,v = self.hsv
            h = 0.5 + 0.5*atan2(y-self.y0, self.x0-x)/pi
            return (h,s,v), True
        elif area == AREA_SAMPLE:
            a = pi+atan2(y-self.y0, self.x0-x)
            for i,a1 in enumerate(self.angles):
                if a1-2*pi/CIRCLE_N < a < a1:
                    clr = self.simple_colors[i]
                    return clr, False
            return None, False
        elif area == AREA_SQUARE:
            h,s,v = self.hsv
            s = (x-self.x0+self.m)/(2*self.m)
            v = (y-self.y0+self.m)/(2*self.m)
            return (h,s,1-v), True
        return None, False

    def draw_square(self,width,height,radius):
        img = cairo.ImageSurface(cairo.FORMAT_ARGB32, width,height)
        cr = cairo.Context(img)
        m = radius/sqrt(2.0)

        # Slightly darker border for the colour area
        sw = max(0.75*self.stroke_width, 3.0)
        cr.set_source_rgb(*NEUTRAL_DARK_GREY)
        cr.rectangle(self.x0-m-sw, self.y0-m-sw, 2*(m+sw), 2*(m+sw))
        cr.fill()

        h,s,v = self.hsv
        ds = 2*m*CSTEP
        v = 0.0
        x1 = self.x0-m
        x2 = self.x0+m
        y = self.y0-m
        while v < 1.0:
            g = cairo.LinearGradient(x1,y,x2,y)
            g.add_color_stop_rgb(0.0, *hsv_to_rgb(h,0.0,1.0-v))
            g.add_color_stop_rgb(1.0, *hsv_to_rgb(h,1.0,1.0-v))
            cr.set_source(g)
            cr.rectangle(x1,y, 2*m, ds)
            cr.fill_preserve()
            cr.stroke()
            y += ds
            v += CSTEP
        h,s,v = self.hsv
        x = self.x0-m + s*2*m
        y = self.y0-m + (1-v)*2*m
        cr.set_source_rgb(*hsv_to_rgb(1-h,1-s,1-v))
        cr.arc(x,y, 3.0, 0.0, 2*pi)
        cr.stroke()

        return img

    def small_circle(self, cr, size, color, angle, rad):
        x0 = self.x0 + rad*cos(angle)
        y0 = self.y0 + rad*sin(angle)
        cr.set_source_rgb(*color)
        cr.arc(x0,y0,size/2., 0, 2*pi)
        cr.fill()

    def small_triangle(self, cr, size, color, angle, rad):
        x0 = self.x0 + rad*cos(angle)
        y0 = self.y0 + rad*sin(angle)
        x1,y1 = x0, y0 - size*sq32
        x2,y2 = x0 + 0.5*size, y0 + size*sq36
        x3,y3 = x0 - 0.5*size, y0 + size*sq36
        cr.set_source_rgb(*color)
        cr.move_to(x1,y1)
        cr.line_to(x2,y2)
        cr.line_to(x3,y3)
        cr.line_to(x1,y1)
        cr.fill()

    def small_triangle_down(self, cr, size, color, angle, rad):
        x0 = self.x0 + rad*cos(angle)
        y0 = self.y0 + rad*sin(angle)
        x1,y1 = x0, y0 + size*sq32
        x2,y2 = x0 + 0.5*size, y0 - size*sq36
        x3,y3 = x0 - 0.5*size, y0 - size*sq36
        cr.set_source_rgb(*color)
        cr.move_to(x1,y1)
        cr.line_to(x2,y2)
        cr.line_to(x3,y3)
        cr.line_to(x1,y1)
        cr.fill()

    def small_square(self, cr, size, color, angle, rad):
        x0 = self.x0 + rad*cos(angle)
        y0 = self.y0 + rad*sin(angle)
        x1,y1 = x0+size*sq22, y0+size*sq22
        x2,y2 = x0+size*sq22, y0-size*sq22
        x3,y3 = x0-size*sq22, y0-size*sq22
        x4,y4 = x0-size*sq22, y0+size*sq22
        cr.set_source_rgb(*color)
        cr.move_to(x1,y1)
        cr.line_to(x2,y2)
        cr.line_to(x3,y3)
        cr.line_to(x4,y4)
        cr.fill()

    def small_rect(self, cr, size, color, angle, rad):
        x0 = self.x0 + rad*cos(angle)
        y0 = self.y0 + rad*sin(angle)
        x1,y1 = x0+size*sq22, y0+size*0.3
        x2,y2 = x0+size*sq22, y0-size*0.3
        x3,y3 = x0-size*sq22, y0-size*0.3
        x4,y4 = x0-size*sq22, y0+size*0.3
        cr.set_source_rgb(*color)
        cr.move_to(x1,y1)
        cr.line_to(x2,y2)
        cr.line_to(x3,y3)
        cr.line_to(x4,y4)
        cr.fill()

    def small_rect_vert(self, cr, size, color, angle, rad):
        x0 = self.x0 + rad*cos(angle)
        y0 = self.y0 + rad*sin(angle)
        x1,y1 = x0+size*0.3, y0+size*sq22
        x2,y2 = x0+size*0.3, y0-size*sq22
        x3,y3 = x0-size*0.3, y0-size*sq22
        x4,y4 = x0-size*0.3, y0+size*sq22
        cr.set_source_rgb(*color)
        cr.move_to(x1,y1)
        cr.line_to(x2,y2)
        cr.line_to(x3,y3)
        cr.line_to(x4,y4)
        cr.fill()

    def inv(self, rgb):
        r,g,b = rgb
        return 1-r,1-g,1-b

    def draw_central_fill(self, cr, r):
        """Draws a neutral-coloured grey circle of the specified radius in the central area."""
        cr.set_source_rgb(*NEUTRAL_MID_GREY)
        cr.arc(self.x0, self.y0, r, 0, 2*pi)
        cr.fill()

    def draw_harmony_ring(self,width,height):
        """Draws the harmony ring if any colour harmonies are visible."""
        if not self.window:
            return
        points_size = min(width,height)/27.
        img = cairo.ImageSurface(cairo.FORMAT_ARGB32, width,height)
        cr = cairo.Context(img)
        if not self.has_harmonies_visible():
            self.draw_central_fill(cr, self.r2)
            return img
        h,s,v = self.hsv
        a = h*2*pi
        self.angles = []
        self.simple_colors = []
        cr.set_line_width(0.75*self.stroke_width)
        self.samples = [self.hsv]
        for i in range(int(CIRCLE_N)):
            c1 = c = h + i/CIRCLE_N
            if c1 > 1:
                c1 -= 1
            clr = hsv_to_rgb(c1,s,v)
            hsv = c1,s,v
            self.simple_colors.append(clr)
            delta = c1 - h
            an = -c1*2*pi
            self.angles.append(-an+pi/CIRCLE_N)
            a1 = an-pi/CIRCLE_N
            a2 = an+pi/CIRCLE_N
            cr.new_path()
            cr.set_source_rgb(*clr)
            cr.move_to(self.x0,self.y0)
            cr.arc(self.x0, self.y0, self.r2, a1, a2)
            cr.line_to(self.x0,self.y0)
            cr.fill_preserve()
            cr.set_source_rgb(*NEUTRAL_DARK_GREY) # "lines" between samples
            cr.stroke()
            # Indicate harmonic colors
            if self.app.preferences.get("colorsampler.triadic", False) and i%(CIRCLE_N/3)==0:
                self.small_triangle(cr, points_size, self.inv(clr), an, (self.r2+self.rd)/2)
                try_put(self.samples, hsv)
            if self.app.preferences.get("colorsampler.complementary", False) and i%(CIRCLE_N/2)==0:
                self.small_circle(cr, points_size, self.inv(clr), an, (self.r2+self.rd)/2)
                try_put(self.samples, hsv)
            if self.app.preferences.get("colorsampler.square", False) and i%(CIRCLE_N/4)==0:
                self.small_square(cr, points_size, self.inv(clr), an, (self.r2+self.rd)/2)
                try_put(self.samples, hsv)
# FIXME: should this harmonies be expressed in terms of CIRCLE_N?
            if self.app.preferences.get("colorsampler.double_comp", False) and i in [0,2,6,8]:
                self.small_rect_vert(cr, points_size, self.inv(clr), an, (self.r2+self.rd)/2)
                try_put(self.samples, hsv)
            if self.app.preferences.get("colorsampler.split_comp", False) and i in [0,5,7]:
                self.small_triangle_down(cr, points_size, self.inv(clr), an, (self.r2+self.rd)/2)
                try_put(self.samples, hsv)
            if self.app.preferences.get("colorsampler.analogous", False) and i in [0,1,CIRCLE_N-1]:
                self.small_rect(cr, points_size, self.inv(clr), an, (self.r2+self.rd)/2)
                try_put(self.samples, hsv)
        # Fill the centre
        self.draw_central_fill(cr, self.rd)
        # And an inner thin line
        cr.set_source_rgb(*NEUTRAL_DARK_GREY)
        cr.arc(self.x0, self.y0, self.rd, 0, 2*pi)
        cr.stroke()
        return img

    def draw_circle_indicator(self, cr):
        """Draws the indicator which shows the current hue on the outer ring."""
        h, s, v = self.hsv
        a = h*2*pi
        x1 = self.x0 + self.r2*cos(-a)
        y1 = self.y0 + self.r2*sin(-a)
        x2 = self.x0 + self.r3*cos(-a)
        y2 = self.y0 + self.r3*sin(-a)
        self.last_line = x1,y1, x2,y2, h
        cr.set_line_width(0.8*self.stroke_width)
        cr.set_source_rgb(0, 0, 0)
        cr.move_to(x1,y1)
        cr.line_to(x2,y2)
        cr.stroke()

    def draw_circles(self, cr):
        """Draws two grey lines just inside and outside the outer hue ring."""
        cr.set_line_width(0.75*self.stroke_width)
        cr.set_source_rgb(*NEUTRAL_DARK_GREY)
        cr.arc(self.x0,self.y0, self.r2, 0, 2*pi)
        cr.stroke()
        cr.arc(self.x0,self.y0, self.r3, 0, 2*pi)
        cr.stroke()
        
    def draw_circle(self, w, h):
        if self.circle_img:
            return self.circle_img
        img = cairo.ImageSurface(cairo.FORMAT_ARGB32, w,h)
        cr = cairo.Context(img)
        cr.set_line_width(0.75*self.stroke_width)
        a1 = 0.0
        while a1 < 2*pi:
            clr = hsv_to_rgb(a1/(2*pi), 1.0, 1.0)
            x1,y1,x2,y2 = self.calc_line(a1)
            a1 += CSTEP
            cr.set_source_rgb(*clr)
            cr.move_to(x1,y1)
            cr.line_to(x2,y2)
            cr.stroke()
        self.circle_img = img
        return img

    def draw_comparison_semicircles(self, cr):
        sq2 = sqrt(2.0)
        M = self.M/2.0
        r = (sq2-1)*M/(2*sq2)
        x0 = self.x0+M-r
        y0 = self.y0+M-r
        a0 = pi/4

        prev_col = self.get_previous_color()
        curr_col = self.color

        cr.set_source_rgb(*NEUTRAL_MID_GREY)
        cr.arc(x0, y0, 0.9*r, a0, a0+(2*pi))
        cr.fill()

        cr.set_source_rgb(*prev_col)
        cr.arc(x0, y0, 0.8*r, a0, a0+(2*pi))
        cr.fill()

        cr.set_source_rgb(*curr_col)
        cr.arc(x0, y0, 0.8*r, a0+(pi/2), a0+(3*pi/2))
        cr.fill()

    def draw(self,w,event):
        if not self.window:
            return
        cr = self.window.cairo_create()
        cr.set_source_surface(self.draw_circle(self.w,self.h))
        cr.paint()
        cr.set_source_surface(self.draw_harmony_ring(self.w,self.h))
        cr.paint()
        self.draw_circle_indicator(cr)
        self.draw_circles(cr)
        cr.set_source_surface(self.draw_square(self.w, self.h, self.rd*0.95))
        cr.paint()
        self.draw_comparison_semicircles(cr)

class VSelector(GColorSelector):
    def __init__(self, app, color=(1,0,0), height=16):
        GColorSelector.__init__(self,app)
        self.color = color
        self.hsv = rgb_to_hsv(*color)
        self.set_size_request(height*2,height)

    def get_color_at(self, x,y):
        h,s,v = self.hsv
        v = clamp(x/self.w, 0.0, 1.0)
        return (h,s,v), True

    def move(self, x,y):
        self.select_color_at(x,y)

    def draw_gradient(self,cr, start,end, hsv=True):
        if hsv:
            clr1 = hsv_to_rgb(*start)
            clr2 = hsv_to_rgb(*end)
        else:
            clr1 = start
            clr2 = end
        g = cairo.LinearGradient(0,0,self.w,self.h)
        g.add_color_stop_rgb(0.0, *clr1)
        g.add_color_stop_rgb(1.0, *clr2)
        cr.set_source(g)
        cr.rectangle(0,0,self.w,self.h)
        cr.fill()

    def draw_line_at(self, cr, x):
        cr.set_source_rgb(0,0,0)
        cr.move_to(x,0)
        cr.line_to(x, self.h)
        cr.stroke()

    def draw(self,w, event):
        if not self.window:
            return
        cr = self.window.cairo_create()
        h,s,v = self.hsv
        self.draw_gradient(cr, (h,s,0.), (h,s,1.))

        x1 = v*self.w
        self.draw_line_at(cr, x1)

class HSelector(VSelector):
    def get_color_at(self,x,y):
        h,s,v = self.hsv
        h = clamp(x/self.w, 0.0, 1.0)
        return (h,s,v), True
    
    def draw(self,w, event):
        if not self.window:
            return
        cr = self.window.cairo_create()
        h,s,v = self.hsv
        dx = self.w*CSTEP
        x = 0
        h1 = 0.
        while h1 < 1:
            cr.set_source_rgb(*hsv_to_rgb(h1,s,v))
            cr.rectangle(x,0,dx,self.h)
            cr.fill_preserve()
            cr.stroke()
            h1 += CSTEP
            x += dx
        x1 = h*self.w
        self.draw_line_at(cr, x1)

class SSelector(VSelector):
    def get_color_at(self, x,y):
        h,s,v = self.hsv
        s = clamp(x/self.w, 0.0, 1.0)
        return (h,s,v), True

    def draw(self,w, event):
        if not self.window:
            return
        cr = self.window.cairo_create()
        h,s,v = self.hsv
        self.draw_gradient(cr, (h,0.,v), (h,1.,v))

        x1 = s*self.w
        self.draw_line_at(cr, x1)

class RSelector(VSelector):
    def get_color_at(self,x,y):
        r,g,b = self.color
        r = clamp(x/self.w, 0.0, 1.0)
        return (r,g,b), False
    
    def draw(self,w, event):
        if not self.window:
            return
        cr = self.window.cairo_create()
        r,g,b = self.color
        self.draw_gradient(cr, (0.,g,b),(1.,g,b), hsv=False)
        x1 = r*self.w
        self.draw_line_at(cr,x1)

class GSelector(VSelector):
    def get_color_at(self,x,y):
        r,g,b = self.color
        g = clamp(x/self.w, 0.0, 1.0)
        return (r,g,b), False
    
    def draw(self,w, event):
        if not self.window:
            return
        cr = self.window.cairo_create()
        r,g,b = self.color
        self.draw_gradient(cr, (r,0.,b),(r,1.,b), hsv=False)
        x1 = g*self.w
        self.draw_line_at(cr,x1)

class BSelector(VSelector):
    def get_color_at(self,x,y):
        r,g,b = self.color
        b = clamp(x/self.w, 0.0, 1.0)
        return (r,g,b), False
    
    def draw(self,w, event):
        if not self.window:
            return
        cr = self.window.cairo_create()
        r,g,b = self.color
        self.draw_gradient(cr, (r,g,0.),(r,g,1.), hsv=False)
        x1 = b*self.w
        self.draw_line_at(cr,x1)

def make_spin(min,max, changed_cb):
    adj = gtk.Adjustment(0,min,max, 1,10)
    btn = gtk.SpinButton(adj)
    btn.connect('value-changed', changed_cb)
    btn.set_sensitive(False)
    return btn

class HSVSelector(gtk.VBox):
    def __init__(self, app, color=(1.,0.,0)):
        gtk.VBox.__init__(self)
        self.color = color
        self.hsv = rgb_to_hsv(*color)
        self.atomic = False

        hbox = gtk.HBox()
        self.hsel = hsel = HSelector(app, color)
        hsel.on_select = self.user_selected_color
        self.hspin = hspin = make_spin(0,359, self.hue_change)
        hbox.pack_start(hsel, expand=True)
        hbox.pack_start(hspin, expand=False)

        sbox = gtk.HBox()
        self.ssel = ssel = SSelector(app, color)
        ssel.on_select = self.user_selected_color
        self.sspin = sspin = make_spin(0,100, self.sat_change)
        sbox.pack_start(ssel, expand=True)
        sbox.pack_start(sspin, expand=False)

        vbox = gtk.HBox()
        self.vsel = vsel = VSelector(app, color)
        vsel.on_select = self.user_selected_color
        self.vspin = vspin = make_spin(0,100, self.val_change)
        vbox.pack_start(vsel, expand=True)
        vbox.pack_start(vspin, expand=False)
        
        self.pack_start(hbox, expand=False)
        self.pack_start(sbox, expand=False)
        self.pack_start(vbox, expand=False)

        self.set_tooltip_text(_("Change Hue, Saturation and Value"))

    def user_selected_color(self, hsv):
        self.set_color(hsv)
        self.on_select(hsv)

    def set_color(self, hsv):
        self.atomic = True
        self.hsv = h,s,v = hsv
        self.hspin.set_value(h*359)
        self.sspin.set_value(s*100)
        self.vspin.set_value(v*100)
        self.hsel.set_color(hsv)
        self.ssel.set_color(hsv)
        self.vsel.set_color(hsv)
        self.atomic = False
        self.color = hsv_to_rgb(*hsv)

    def on_select(self, color):
        pass

    def hue_change(self, spin):
        if self.atomic:
            return
        h,s,v = self.hsv
        self.set_color((spin.get_value()/359., s,v))

    def sat_change(self, spin):
        if self.atomic:
            return
        h,s,v = self.hsv
        self.set_color((h, spin.get_value()/100., v))

    def val_change(self, spin):
        if self.atomic:
            return
        h,s,v = self.hsv
        self.set_color((h,s, spin.get_value()/100.))

class RGBSelector(gtk.VBox):
    def __init__(self, app, color=(1.,0.,0)):
        gtk.VBox.__init__(self)
        self.color = color
        self.hsv = rgb_to_hsv(*color)
        self.atomic = False

        rbox = gtk.HBox()
        self.rsel = rsel = RSelector(app, color)
        rsel.on_select = self.user_selected_color
        self.rspin = rspin = make_spin(0,255, self.r_change)
        rbox.pack_start(rsel, expand=True)
        rbox.pack_start(rspin, expand=False)

        gbox = gtk.HBox()
        self.gsel = gsel = GSelector(app, color)
        gsel.on_select = self.user_selected_color
        self.gspin = gspin = make_spin(0,255, self.g_change)
        gbox.pack_start(gsel, expand=True)
        gbox.pack_start(gspin, expand=False)

        bbox = gtk.HBox()
        self.bsel = bsel = BSelector(app, color)
        bsel.on_select = self.user_selected_color
        self.bspin = bspin = make_spin(0,255, self.b_change)
        bbox.pack_start(bsel, expand=True)
        bbox.pack_start(bspin, expand=False)
        
        self.pack_start(rbox, expand=False)
        self.pack_start(gbox, expand=False)
        self.pack_start(bbox, expand=False)

        self.set_tooltip_text(_("Change Red, Green and Blue components"))

    def user_selected_color(self, hsv):
        self.set_color(hsv)
        self.on_select(hsv)

    def set_color(self, hsv):
        self.atomic = True
        self.color = r,g,b = hsv_to_rgb(*hsv)
        self.rspin.set_value(r*255)
        self.gspin.set_value(g*255)
        self.bspin.set_value(b*255)
        self.rsel.set_color(hsv)
        self.gsel.set_color(hsv)
        self.bsel.set_color(hsv)
        self.atomic = False
        self.hsv = hsv

    def on_select(self, hsv):
        pass

    def r_change(self, spin):
        if self.atomic:
            return
        r,g,b = self.color
        self.set_color(rgb_to_hsv(spin.get_value()/255., g,b))

    def g_change(self, spin):
        if self.atomic:
            return
        r,g,b = self.color
        self.set_color(rgb_to_hsv(r, spin.get_value()/255., b))

    def b_change(self, spin):
        if self.atomic:
            return
        r,g,b = self.color
        self.set_color(rgb_to_hsv(r,g, spin.get_value()/255.))

class Selector(gtk.VBox):

    CIRCLE_MIN_SIZE = (150, 150)

    def __init__(self, app):
        gtk.VBox.__init__(self)
        self.app = app
        self.recent = RecentColors(app)
        self.circle = CircleSelector(app)
        self.circle.set_size_request(*self.CIRCLE_MIN_SIZE)

        # The colour circle seems to need its own window to draw sanely when
        # packed into a Tool wrapper, which probably indicates that something's
        # amiss somewhere.
        evbox = gtk.EventBox()
        evbox.add(self.circle)
        self.pack_start(evbox, expand=True)

        self.rgb_selector = RGBSelector(app)
        self.hsv_selector = HSVSelector(app)
        self.rgb_selector.on_select = self.rgb_selected
        self.hsv_selector.on_select = self.hsv_selected
        nb = gtk.Notebook()
        nb.append_page(self.rgb_selector, gtk.Label(_('RGB')))
        nb.append_page(self.hsv_selector, gtk.Label(_('HSV')))

        # Colour history
        self.exp_history = expander = ElasticExpander(_('Colors history'))
        expander.set_spacing(6)
        expander.add(self.recent)
        expander.connect("notify::expanded", self.expander_expanded_cb, 'history')
        self.pack_start(expander, expand=False)

        # Colour details
        self.exp_details = expander = ElasticExpander(_('Details'))
        expander.set_spacing(6)
        expander.add(nb)
        expander.connect("notify::expanded", self.expander_expanded_cb, 'details')
        self.pack_start(expander, expand=False)

        # Colour scheme harmonies
        def harmony_checkbox(attr, label, tooltip=None):
            cb = gtk.CheckButton(label)
            pref = "colorsampler.%s" % (attr,)
            cb.set_active(self.app.preferences.get(pref, False))
            cb.connect('toggled', self.harmony_toggled, attr)
            if tooltip is not None:
                cb.set_tooltip_text(tooltip)
            vbox2.pack_start(cb, expand=False)

        self.exp_config = expander = ElasticExpander(_('Harmonies'))
        vbox2 = gtk.VBox()
        harmony_checkbox('analogous', _('Analogous'), _("Three nearby hues on the color wheel.\nOften found in nature, frequently pleasing to the eye."))
        harmony_checkbox('complementary', _('Complementary color'), _("Two opposite hues on the color wheel.\nVibrant, and maximally contrasting."))
        harmony_checkbox('split_comp', _('Split complementary'), _("Two hues next to the current hue's complement.\nContrasting, but adds a possibly pleasing harmony."))
        harmony_checkbox('double_comp', _('Double complementary'), _("Four hues in two complementary pairs."))
        harmony_checkbox('square', _('Square'), _("Four equally-spaced hues"))
        harmony_checkbox('triadic', _('Triadic'), _("Three equally-spaced hues.\nVibrant with equal tension."))

        vbox3 = gtk.VBox()
        cb_sv = gtk.CheckButton(_('Change value/saturation'))
        cb_sv.set_active(True)
        cb_sv.connect('toggled', self.toggle_blend, 'value')
        cb_opposite = gtk.CheckButton(_('Blend each color to opposite'))
        cb_opposite.connect('toggled', self.toggle_blend, 'opposite')
        cb_neg = gtk.CheckButton(_('Blend each color to negative'))
        cb_neg.connect('toggled', self.toggle_blend, 'negative')
        vbox3.pack_start(cb_sv, expand=False)
        vbox3.pack_start(cb_opposite, expand=False)
        vbox3.pack_start(cb_neg, expand=False)
        expander.add(vbox2)
        expander.connect("notify::expanded", self.expander_expanded_cb, 'harmonies')
        self.pack_start(expander, expand=False)

        self.circle.on_select = self.hue_selected
        self.circle.get_previous_color = self.previous_color
        self.recent.on_select = self.recent_selected
        self.widgets = [self.circle, self.rgb_selector, self.hsv_selector, self.recent]

        self.value_blends = True
        self.opposite_blends = False
        self.negative_blends = False

        self.expander_prefs_loaded = False
        self.connect("show", self.show_cb)

    def toggle_blend(self, checkbox, name):
        attr = name+'_blends'
        setattr(self, attr, not getattr(self, attr))

    def harmony_toggled(self, checkbox, attr):
        pref = "colorsampler.%s" % (attr,)
        self.app.preferences[pref] = checkbox.get_active()
        self.circle.configure_calc()
        self.queue_draw()

    def previous_color(self):
        return self.app.ch.last_color

    def set_color(self, hsv, exclude=None):
        for w in self.widgets:
            if w is not exclude:
                w.set_color(hsv)
        self.color = hsv_to_rgb(*hsv)
        self.hsv = hsv
        self.on_select(hsv)

    def rgb_selected(self, hsv):
        self.set_color(hsv, exclude=self.rgb_selector)

    def hsv_selected(self, hsv):
        self.set_color(hsv, exclude=self.hsv_selector)

    def hue_selected(self, hsv):
        self.set_color(hsv, exclude=self.circle)

    def recent_selected(self, hsv):
        self.set_color(hsv, exclude=self.recent)

    def on_select(self, hsv):
        pass

    def expander_expanded_cb(self, expander, junk, cfg_stem):
        # Save the expander state
        if not self.expander_prefs_loaded:
            return
        self.app.preferences['colorsampler.%s_expanded' % cfg_stem] \
          = bool(expander.get_expanded())

    def show_cb(self, widget):
        # Restore expander state from prefs.
        # Use "show", not "map", so that it fires before we have a window
        assert not self.window
        assert not self.expander_prefs_loaded
        # If we wait until then, sidebar positions will drift as a result.
        if self.app.preferences.get("colorsampler.history_expanded", False):
            self.exp_history.set_expanded(True)
        if self.app.preferences.get("colorsampler.details_expanded", False):
            self.exp_details.set_expanded(True)
        if self.app.preferences.get("colorsampler.harmonies_expanded", False):
            self.exp_config.set_expanded(True)
        self.expander_prefs_loaded = True


class ToolWidget (Selector):

    tool_widget_title = _("MyPaint color selector")

    def __init__(self, app):
        Selector.__init__(self, app)

        self.app.brush.observers.append(self.brush_modified_cb)
        self.stop_callback = False

        # The first callback notification happens before the window is initialized
        self.set_color(app.brush.get_color_hsv())

    def brush_modified_cb(self, settings):
        if not settings.intersection(('color_h', 'color_s', 'color_v')):
            return
        if self.stop_callback:
            return
        self.stop_callback = True
        hsv = self.app.brush.get_color_hsv()
        self.set_color(hsv)
        self.stop_callback = False

    def on_select(self, hsv):
        if self.stop_callback:
            return
        self.app.brush.set_color_hsv(hsv)


class Window(windowing.SubWindow):
    def __init__(self,app):
        windowing.SubWindow.__init__(self, app)
        self.set_title(_('MyPaint color selector'))
        self.set_role('Color selector')
        self.selector = Selector(app, self)
        self.selector.on_select = self.on_select
        self.exp_history = self.selector.exp_history
        self.exp_details = self.selector.exp_details
        self.exp_config = self.selector.exp_config

        self.add(self.selector)
        self.app.brush.observers.append(self.brush_modified_cb)
        self.stop_callback = False

        # The first callback notification happens before the window is initialized
        self.selector.set_color(app.brush.get_color_hsv())

    def brush_modified_cb(self, settings):
        if not settings.intersection(('color_h', 'color_s', 'color_v')):
            return
        if self.stop_callback:
            return
        self.stop_callback = True
        hsv = self.app.brush.get_color_hsv()
        self.selector.set_color(hsv)
        self.stop_callback = False

    def on_select(self, hsv):
        if self.stop_callback:
            return
        self.app.brush.set_color_hsv(hsv)
