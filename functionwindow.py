"window to model a single brush property function"
import gtk
import brush

class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app
        self.app.brush_selected_callbacks.append(self.brush_selected_cb)
        self.add_accel_group(self.app.accel_group)

        self.set_title('Curves')

        inputs = ['pressure', 'speed', 'speed2']
        table = gtk.Table(1, 2*len(inputs))
        self.add(table)

        self.tooltips = gtk.Tooltips()

        pos = 0
        for input in inputs:
            c =  CurveWidget()

            l = gtk.Label(input)
            table.attach(l, 0, 1, pos, pos+1, gtk.FILL, gtk.FILL, 5, 0)
            pos += 1
            table.attach(c, 0, 1, pos, pos+1, gtk.EXPAND | gtk.FILL, gtk.EXPAND | gtk.FILL, 5, 0)
            pos += 1


        self.set_size_request(450, 500)

    def brush_selected_cb(self, brush_selected):
        pass
        #for s in brush.brushsettings:
        #    self.adj[s.index].set_value(self.app.brush.get_setting(s.index))



RADIUS = 4
class CurveWidget(gtk.DrawingArea):
    "modify a nonlinear curve"
    def __init__(self):
        gtk.DrawingArea.__init__(self)
        self.points = [(0.0, 0.5), (1.0, 1.0)]
        self.maxpoints = 5
        self.grabbed = None

        self.connect("expose-event", self.expose_cb)
        self.connect("button-press-event", self.button_press_cb)
        self.connect("button-release-event", self.button_release_cb)
        self.connect("motion-notify-event", self.motion_notify_cb)
	self.set_events(gtk.gdk.EXPOSURE_MASK |
                        gtk.gdk.BUTTON_PRESS_MASK |
                        gtk.gdk.BUTTON_RELEASE_MASK |
                        gtk.gdk.POINTER_MOTION_MASK
                        )

    def eventpoint(self, event_x, event_y):
        # FIXME: very ugly; and code duplication, see expose_cb
        width, height = self.window.get_size()
        width  -= 2*RADIUS
        height -= 2*RADIUS
        width  = width / 4 * 4
        height = height / 4 * 4
        x, y = event_x, event_y
        x -= RADIUS
        y -= RADIUS
        x = float(x) / width
        y = float(y) / height
        return (x, y)

    def button_press_cb(self, widget, event):
        if not event.button == 1: return
        x, y = self.eventpoint(event.x, event.y)
        nearest = None
        for i in range(len(self.points)):
            px, py = self.points[i]
            dist = abs(px - x)
            if nearest is None or dist < mindist:
                mindist = dist
                nearest = i
        if mindist > 0.05 and len(self.points) < self.maxpoints:
            insertpos = 0
            for i in range(len(self.points)):
                if self.points[i][0] < x:
                    insertpos = i + 1
            if insertpos > 0 and insertpos < len(self.points):
                self.points.insert(insertpos, (x, y))
                nearest = insertpos
                self.queue_draw()

        if nearest == 0: 
            # first point cannot be grabbed
            display = gtk.gdk.display_get_default()
            display.beep()
        else:
            assert self.grabbed is None
            self.grabbed = nearest

    def button_release_cb(self, widget, event):
        if not event.button == 1: return
        if self.grabbed:
            i = self.grabbed
            if self.points[i] is None:
                self.points.pop(i)
        self.grabbed = None
            
    def motion_notify_cb(self, widget, event):
        if self.grabbed is None: return
        x, y = self.eventpoint(event.x, event.y)
        i = self.grabbed
        assert i > 0
        if i == len(self.points)-1:
            x = 1.0 # right point stays right
            # and cannot be moved out
            out = False
        else:
            out = False
            if y > 1.1 or y < -0.1: out = True
            leftbound  = self.points[i-1][0]
            rightbound = self.points[i+1][0]
            if x <= leftbound or x >= rightbound: out = True
        if out:
            self.points[i] = None
        else:
            if y > 1.0: y = 1.0
            if y < 0.0: y = 0.0
            if y > 0.47 and y < 0.53: y = 0.5 # snap
            self.points[i] = (x, y)
        self.queue_draw()


    def expose_cb(self, widget, event):
        window = widget.window
        state = gtk.STATE_NORMAL
        gray  = widget.style.bg_gc[state]
        dark  = widget.style.dark_gc[state]
        black  = widget.style.fg_gc[state]

        width, height = window.get_size()
        window.draw_rectangle(gray, True, 0, 0, width, height)
        width  -= 2*RADIUS
        height -= 2*RADIUS
        width  = width / 4 * 4
        height = height / 4 * 4
        if width <= 0 or height <= 0: return

        # draw grid lines
        for i in range(5):
            window.draw_line(dark,  RADIUS, i*height/4 + RADIUS,
                             width + RADIUS, i*height/4 + RADIUS)
            window.draw_line(dark, i*width/4 + RADIUS, RADIUS,
                                    i*width/4 + RADIUS, height + RADIUS)
        # draw points
        for p in self.points:
            if p is None: continue
            x1, y1 = p
            x1 = int(x1*width) + RADIUS
            y1 = int(y1*height) + RADIUS
            window.draw_rectangle(black, True, x1-RADIUS-1, y1-RADIUS-1, 2*RADIUS+1, 2*RADIUS+1)
            if p is not self.points[0]:
                window.draw_line(black, x0, y0, x1, y1)
            x0, y0 = x1, y1

        return True
