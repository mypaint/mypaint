from gettext import gettext as _
import gtk
gdk = gtk.gdk
from lib.helpers import rgb_to_hsv, hsv_to_rgb
from math import pi, sin, cos, sqrt, atan2
import struct

CSTEP=0.007
PADDING=4

class GColorSelector(gtk.DrawingArea):
    def __init__(self):
        gtk.DrawingArea.__init__(self)
        self.color = (1,0,0)
        self.hsv = (0,1,1)
        self.connect('expose-event', self.draw)
        self.set_events(gdk.BUTTON_PRESS_MASK |
                        gdk.BUTTON_RELEASE_MASK |
                        gdk.POINTER_MOTION_MASK |
                        gdk.ENTER_NOTIFY |
                        gdk.LEAVE_NOTIFY |
                        gdk.DROP_FINISHED |
                        gdk.DROP_START |
                        gdk.DRAG_STATUS)
        self.connect('button-press-event', self.on_button_press)
        self.connect('button-release-event', self.on_button_release)
        self.connect('configure-event', self.on_configure)
        self.connect('motion-notify-event', self.motion)
        self.connect('drag_data_get', self.drag_get)
        self.button_pressed = False
        self.do_select = True

    def motion(self,w, event):
        if not self.button_pressed:
            return
        d = sqrt((event.x-self.press_x)**2 + (event.y-self.press_y)**2)
        if d > 20:
            self.do_select = False
            self.drag_begin([("application/x-color",0,80)], gdk.ACTION_COPY, 1, event)

    def drag_get(self, widget, context, selection, targetType, eventTime):
        r,g,b = self.color
        r = min(int(r*65536), 65535)
        g = min(int(g*65536), 65535)
        b = min(int(b*65536), 65535)
        clrs = struct.pack('HHHH',r,g,b,0)
        selection.set(selection.target, 8, clrs)

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

    def select_color_at(self, x,y):
        pass

    def set_color(self, color):
        self.color = color
        self.hsv = rgb_to_hsv(*color)
        self.queue_draw()

    def get_color(self):
        return self.color

    def on_button_press(self,w, event):
        self.button_pressed = True
        self.do_select = True
        self.press_x = event.x
        self.press_y = event.y

    def on_button_release(self,w, event):
        if self.button_pressed and self.do_select:
            self.select_color_at(event.x,event.y)
            self.on_select(self.color)
        self.button_pressed = False

    def draw(self,w,event):
        if not self.window:
            return
        cr = self.window.cairo_create()
        cr.set_source_rgb(*self.color)
        cr.rectangle(PADDING,PADDING, self.w-2*PADDING, self.h-2*PADDING)
        cr.fill()

class RectSlot(GColorSelector):
    def __init__(self,color=(1,1,1),size=48):
        GColorSelector.__init__(self)
        self.color = color
        self.set_size_request(size,size)

class RecentColors(gtk.HBox):
    def __init__(self, N=5):
        gtk.HBox.__init__(self)
        self.set_border_width(4)
        self.N = N
        self.slots = []
        self.colors = []
        for i in range(N):
            slot = RectSlot()
            slot.on_select = self.slot_selected
            self.pack_start(slot, expand=True)
            self.slots.append(slot)
            self.colors.append(slot.color)
        self.show_all()
        self.last_color = (1.0,1.0,1.0)

    def slot_selected(self,color):
        self.on_select(color)

    def on_select(self,color):
        pass

    def set_color(self, color):
        eps = 0.0001
        r1,g1,b1 = color
        r2,g2,b2 = self.last_color
        if abs(r1-r2)<eps and abs(g1-g2)<eps and abs(b1-b2)<eps:
            return
        self.last_color = color
        if color in self.colors:
            self.colors.remove(color)
        self.colors.insert(0, color)
        if len(self.colors) > self.N:
            self.colors = self.colors[:-1]
        for color,slot in zip(self.colors, self.slots):
            slot.set_color(color)

CIRCLE_N = 8.0

class CircleSelector(GColorSelector):
    def __init__(self, color=(1,0,0)):
        GColorSelector.__init__(self)
        self.color = color
        self.hsv = rgb_to_hsv(*color)
        self.last_line = None
        self.calc_colors()

    def get_previous_color(self,n):
        return self.color

    def calc_line(self, angle):
        x1 = self.x0 + self.r2*cos(-angle)
        y1 = self.y0 + self.r2*sin(-angle)
        x2 = self.x0 + self.r3*cos(-angle)
        y2 = self.y0 + self.r3*sin(-angle)
        return x1,y1,x2,y2

    def calc_circle(self):
        self.circle = []
        a1 = 0.0
        while a1 < 2*pi:
            x1,y1,x2,y2 = self.calc_line(a1)
            self.circle.append((x1,y1,x2,y2))
            a1 += CSTEP

    def calc_colors(self):
        self.colors = []
        a1 = 0.0
        while a1 < 2*pi:
            clr = hsv_to_rgb(a1/(2*pi), 1.0, 1.0)
            self.colors.append(clr)
            a1 += CSTEP

    def configure_calc(self):
        self.x0 = self.x_rel + self.w/2.0
        self.y0 = self.y_rel + self.h/2.0
        M = min(self.w,self.h)-5
        self.r1 = M/10.0
        self.r2 = 0.36*M
        self.r3 = M/2.0
        self.calc_circle()

    def set_color(self, color, redraw=True):
        self.color = color
        h,s,v = rgb_to_hsv(*color)
        old_h,old_s,old_v = self.hsv
        self.hsv = h,s,v
        if redraw:
            if h!=old_h:
                self.redraw_circle_line()
            if h!=old_h or s!=old_s or v!=old_v:
                self.draw_inside_circle(n=1)

    def select_color_at(self, x,y):
        d = sqrt((x-self.x0)**2 + (y-self.y0)**2)
        if self.r2 < d < self.r3:
            h,s,v = self.hsv
            h = 0.5 + 0.5*atan2(y-self.y0, self.x0-x)/pi
            self.color = hsv_to_rgb(h,s,v)
            self.hsv = (h,s,v)
            self.redraw_circle_line()
            self.draw_inside_circle()
            self.on_select(self.color)
        elif self.r1 < d < self.r2:
            a = pi+atan2(y-self.y0, self.x0-x)
            for i,a1 in enumerate(self.angles):
                if a1-2*pi/CIRCLE_N < a < a1:
                    clr = self.simple_colors[i]
                    self.color = clr
                    self.hsv = rgb_to_hsv(*clr)
                    self.redraw_circle_line()
                    self.draw_inside_circle()
                    self.on_select(self.color)
                    break
        elif d < self.r1 and (x-self.x0) < 0:
            self.hsv = rgb_to_hsv(*self.color)
            self.queue_draw()
            self.on_select(self.color)

    def redraw_circle_line(self):
        if not self.window:
            return
        cr = self.window.cairo_create()
        if not self.last_line:
            x1,y1,x2,y2,h = self.x0+self.r2,self.y0, self.x0+self.r3, self.y0, 0.0
        else:
            x1,y1, x2,y2, h = self.last_line
        cr.set_line_width(5.0)
        cr.set_source_rgb(*hsv_to_rgb(h,1.0,1.0))
        cr.move_to(x1,y1)
        cr.line_to(x2,y2)
        cr.stroke()
        h,s,v = self.hsv
        x1,y1,x2,y2 = self.calc_line(2*pi*h)
        self.last_line = x1,y1,x2,y2, h
        cr.set_line_width(4.0)
        cr.set_source_rgb(0,0,0)
        cr.move_to(x1,y1)
        cr.line_to(x2,y2)
        cr.stroke()
        self.draw_circles(cr)

    def draw_inside_circle(self,n=0,cr=None):
        if not self.window:
            return
        if not cr:
            cr = self.window.cairo_create()
        h,s,v = self.hsv
        a = h*2*pi
        self.angles = []
        self.simple_colors = []
        cr.set_line_width(6.0)
        for i in range(int(CIRCLE_N)):
            c = h + i/CIRCLE_N
            if c > 1:
                c -= 1
            cr.new_path()
            clr = hsv_to_rgb(c,s,v)
            cr.set_source_rgb(*clr)
            self.simple_colors.append(clr)
            an = -c*2*pi
            cr.move_to(self.x0, self.y0)
            self.angles.append(-an+pi/CIRCLE_N)
            cr.arc(self.x0, self.y0, self.r2, an-pi/CIRCLE_N, an+pi/CIRCLE_N)
#             cr.line_to(self.x0, self.y0)
            cr.close_path()
            cr.fill_preserve()
            cr.set_source_rgb(0.5,0.5,0.5)
            cr.stroke()
        x1 = self.x0 + self.r2*cos(-a)
        y1 = self.y0 + self.r2*sin(-a)
        x2 = self.x0 + self.r3*cos(-a)
        y2 = self.y0 + self.r3*sin(-a)
        self.last_line = x1,y1, x2,y2, h
        cr.set_line_width(4.0)
        cr.set_source_rgb(0,0,0)
        cr.move_to(x1,y1)
        cr.line_to(x2,y2)
        cr.stroke()
        cr.set_source_rgb(*self.color)
        cr.arc(self.x0, self.y0, self.r1, -pi/2, pi/2)
        cr.fill()
        cr.set_source_rgb(*self.get_previous_color(n))
        cr.arc(self.x0, self.y0, self.r1, pi/2, 3*pi/2)
        cr.fill()
        cr.arc(self.x0, self.y0, self.r1, 0, 2*pi)
        cr.set_source_rgb(0.5,0.5,0.5)
        cr.stroke()

    def draw_circles(self, cr):
        cr.set_line_width(6.0)
        cr.set_source_rgb(0.5,0.5,0.5)
        cr.arc(self.x0,self.y0, self.r2, 0, 2*pi)
        cr.stroke()
        cr.arc(self.x0,self.y0, self.r3, 0, 2*pi)
        cr.stroke()
        
    def draw(self,w,event):
        if not self.window:
            return
        cr = self.window.cairo_create()
        cr.set_line_width(4.0)
        for (x1,y1,x2,y2),clr in zip(self.circle, self.colors):
            cr.move_to(x1,y1)
            cr.set_source_rgb(*clr)
            cr.line_to(x2,y2)
            cr.stroke()
        self.draw_inside_circle(cr=cr)
        self.draw_circles(cr)

class VSelector(GColorSelector):
    def __init__(self, color=(1,0,0), width=40):
        GColorSelector.__init__(self)
        self.color = color
        self.hsv = rgb_to_hsv(*color)
        self.set_size_request(width,width*2)

    def select_color_at(self, x,y):
        h,s,v = self.hsv
        v = 1-y/self.h
        self.hsv = h,s,v
        self.color = hsv_to_rgb(h,s,v)
        self.queue_draw()
        self.on_select(self.color)

    def draw(self,w, event):
        if not self.window:
            return
        cr = self.window.cairo_create()
        cr.set_line_width(4.0)
        h,s,v = self.hsv
        t = 0.0
        while t < 1.0:
            x1 = 5
            x2 = self.w-10
            y1 = y2 = t*self.h
            cr.set_source_rgb(*hsv_to_rgb(h,s,1-t))
            cr.move_to(x1,y1)
            cr.line_to(x2,y2)
            cr.stroke()
            t += CSTEP
        x1 = 5
        x2 = self.w-10
        y1 = y2 = (1-v)*self.h
        cr.set_source_rgb(0,0,0)
        cr.move_to(x1,y1)
        cr.line_to(x2,y2)
        cr.stroke()

class SSelector(VSelector):
    def select_color_at(self, x,y):
        h,s,v = self.hsv
        s = 1-y/self.h
        self.hsv = h,s,v
        self.color = hsv_to_rgb(h,s,v)
        self.queue_draw()
        self.on_select(self.color)

    def draw(self,w, event):
        if not self.window:
            return
        cr = self.window.cairo_create()
        cr.set_line_width(4.0)
        h,s,v = self.hsv
        t = 0.0
        while t < 1.0:
            x1 = 5
            x2 = self.w-10
            y1 = y2 = t*self.h
            cr.set_source_rgb(*hsv_to_rgb(h,1-t,v))
            cr.move_to(x1,y1)
            cr.line_to(x2,y2)
            cr.stroke()
            t += CSTEP
        x1 = 5
        x2 = self.w-10
        y1 = y2 = (1-s)*self.h
        cr.set_source_rgb(0,0,0)
        cr.move_to(x1,y1)
        cr.line_to(x2,y2)
        cr.stroke()

class RGBSelector(gtk.VBox):
    def __init__(self):
        gtk.VBox.__init__(self)
        self.color = (1,0,0)
        adj = gtk.Adjustment(lower=0,upper=255,step_incr=1,page_incr=10)
        self.r_e = gtk.SpinButton(adj)
        self.g_e = gtk.SpinButton(adj)
        self.b_e = gtk.SpinButton(adj)
        self.r_e.connect('focus-out-event', self.calc_color)
        self.g_e.connect('focus-out-event', self.calc_color)
        self.b_e.connect('focus-out-event', self.calc_color)
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label(_('R: ')), expand=False)
        hbox.pack_start(self.r_e, expand=True)
        self.pack_start(hbox)
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label(_('G: ')), expand=False)
        hbox.pack_start(self.g_e, expand=True)
        self.pack_start(hbox)
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label(_('B: ')), expand=False)
        hbox.pack_start(self.b_e, expand=True)
        self.pack_start(hbox)

    def on_select(self,color):
        pass

    def calc_color(self, spin, event):
        r = self.r_e.get_value()
        g = self.g_e.get_value()
        b = self.b_e.get_value()
        self.color = (r/255.0, g/255.0, b/255.0)
        self.on_select(self.color)

    def set_color(self, color):
        self.color = color
        r,g,b = color
        self.r_e.set_value(r*255)
        self.g_e.set_value(g*255)
        self.b_e.set_value(b*255)

class HSVSelector(gtk.VBox):
    def __init__(self):
        gtk.VBox.__init__(self)
        self.color = (1,0,0)
        self.hsv = (0,1,1)
        adj1 = gtk.Adjustment(lower=0,upper=100,step_incr=1,page_incr=10)
        adj2 = gtk.Adjustment(lower=0,upper=359,step_incr=1,page_incr=10)
        self.h_e = gtk.SpinButton(adj2)
        self.s_e = gtk.SpinButton(adj1)
        self.v_e = gtk.SpinButton(adj1)
        self.h_e.connect('focus-out-event', self.calc_color)
        self.s_e.connect('focus-out-event', self.calc_color)
        self.v_e.connect('focus-out-event', self.calc_color)
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label(_('H: ')), expand=False)
        hbox.pack_start(self.h_e, expand=True)
        self.pack_start(hbox)
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label(_('S: ')), expand=False)
        hbox.pack_start(self.s_e, expand=True)
        self.pack_start(hbox)
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label(_('V: ')), expand=False)
        hbox.pack_start(self.v_e, expand=True)
        self.pack_start(hbox)

    def on_select(self,color):
        pass

    def calc_color(self, spin, event):
        h = self.h_e.get_value()
        s = self.s_e.get_value()
        v = self.v_e.get_value()
        self.hsv = h/359.0, s/100.0, v/100.0
        self.color = hsv_to_rgb(*self.hsv)
        self.on_select(self.color)

    def set_color(self, color):
        self.color = color
        self.hsv = rgb_to_hsv(*color)
        h,s,v = self.hsv
        self.h_e.set_value(h*359)
        self.s_e.set_value(s*100)
        self.v_e.set_value(v*100)

class Selector(gtk.VBox):
    def __init__(self):
        gtk.VBox.__init__(self)
        hbox = gtk.HBox()
        self.pack_start(hbox, expand=True)
        vbox = gtk.VBox()
        hbox.pack_start(vbox,expand=True)
        self.recent = RecentColors()
        self.circle = CircleSelector()
        self.light = VSelector()
        self.saturation = SSelector()
        l_box = gtk.VBox()
        l_box.pack_start(gtk.Label(_('V')), expand=False)
        l_box.pack_start(self.light, expand=True)
        s_box = gtk.VBox()
        s_box.pack_start(gtk.Label(_('S')), expand=False)
        s_box.pack_start(self.saturation, expand=True)
        hbox.pack_start(s_box, expand=False)
        hbox.pack_start(l_box, expand=False)
        vbox.pack_start(self.circle, expand=True)
        self.rgb_selector = RGBSelector()
        self.hsv_selector = HSVSelector()
        self.rgb_selector.on_select = self.rgb_selected
        self.hsv_selector.on_select = self.hsv_selected
        hbox2 = gtk.HBox()
        hbox2.set_spacing(6)
        hbox2.pack_start(self.rgb_selector)
        hbox2.pack_start(self.hsv_selector)
        expander = gtk.Expander(_('Colors history'))
        expander.set_spacing(6)
        expander.add(self.recent)
        self.pack_start(expander, expand=False)
        expander = gtk.Expander(_('Details'))
        expander.set_spacing(6)
        expander.add(hbox2)
        self.pack_start(expander, expand=False)
        self.circle.on_select = self.hue_selected
        self.circle.get_previous_color = self.previous_color
        self.recent.on_select = self.recent_selected
        self.light.on_select = self.light_selected
        self.saturation.on_select = self.saturation_selected
        self.widgets = [self.recent, self.circle, self.light, self.saturation, self.rgb_selector, self.hsv_selector]

        self.connect('drag_data_received',self.drag_data)
        self.drag_dest_set(gtk.DEST_DEFAULT_MOTION | gtk.DEST_DEFAULT_HIGHLIGHT | gtk.DEST_DEFAULT_DROP,
                 [("application/x-color",0,80)],
                 gtk.gdk.ACTION_COPY)

    def previous_color(self,n):
        try:
            return self.recent.colors[n]
        except:
            return (1.0,1.0,1.0)

    def set_color(self,color,exclude=None):
        for w in self.widgets:
            if w is not exclude:
                w.set_color(color)
        self.color = color
        self.on_select(color)

    def drag_data(self, widget, context, x,y, selection, targetType, time):
        r,g,b,a = struct.unpack('HHHH', selection.data)
        clr = (r/65536.0, g/65536.0, b/65536.0)
        self.set_color(clr)

    def rgb_selected(self, color):
        self.set_color(color, exclude=self.rgb_selector)

    def hsv_selected(self, color):
        self.set_color(color, exclude=self.hsv_selector)

    def hue_selected(self, color):
        self.set_color(color, exclude=self.circle)

    def recent_selected(self, color):
        self.set_color(color)

    def light_selected(self,color):
        self.set_color(color, exclude=self.light)

    def saturation_selected(self, color):
        self.set_color(color, exclude=self.saturation)

    def on_select(self,color):
        pass

class Window(gtk.Window):
    def __init__(self,app):
        gtk.Window.__init__(self)
        self.app = app
        self.set_title(_('MyPaint color selector'))
        self.set_role('Color selector')
        self.set_default_size(350,340)
        self.connect('delete-event', self.app.hide_window_cb)
        self.selector = Selector()
        self.selector.on_select = self.on_select
        self.add(self.selector)
        self.app.kbm.add_window(self)
        self.app.brush.settings_observers.append(self.brush_modified_cb)
        self.stop_callback = False

    def brush_modified_cb(self):
        self.stop_callback = True
        self.selector.set_color(self.app.brush.get_color_rgb())
        self.stop_callback = False

    def on_select(self, color):
        if self.stop_callback:
            return
        self.app.colorSelectionWindow.set_color_rgb(color)
        self.app.brush.set_color_rgb(color)

