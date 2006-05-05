"window to model a single brush property function"
import gtk
import brush, brushsettings

class Window(gtk.Window):
    def __init__(self, app, setting, adj):
        gtk.Window.__init__(self)
        self.app = app
        self.app.brush_selected_callbacks.append(self.brush_selected_cb)

        self.set_title(setting.name)
        self.connect('delete-event', self.app.hide_window_cb)
        self.tooltips = gtk.Tooltips()

        scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.POLICY_NEVER, gtk.POLICY_AUTOMATIC)
        self.add(scroll)

        vbox = gtk.VBox()
        vbox.set_spacing(5)
        scroll.add_with_viewport(vbox)
        vbox.set_border_width(5)

        eb = gtk.EventBox()
        l = gtk.Label()
        l.set_alignment(0.0, 0.0)
        l.set_markup('<b><span size="large">%s</span></b>' % setting.name.title())
        self.tooltips.set_tip(eb, setting.tooltip)
        eb.add(l)
        vbox.pack_start(eb, expand=False)

        
        l = gtk.Label()
        l.set_markup('Base Value')
        l.set_alignment(0.0, 0.0)
        l.xpad = 5
        vbox.pack_start(l, expand=False)

        hbox = gtk.HBox()
        vbox.pack_start(hbox, expand=False)
        scale = self.base_value_hscale = gtk.HScale(adj)
        scale.set_digits(2)
        scale.set_draw_value(True)
        scale.set_value_pos(gtk.POS_LEFT)
        hbox.pack_start(scale, expand=True)
        b = gtk.Button("%.1f" % setting.default)
        b.connect('clicked', self.set_fixed_value_clicked_cb, adj, setting.default)
        hbox.pack_start(b, expand=False)

        vbox.pack_start(gtk.HSeparator(), expand=False)

        self.byinputwidgets = []
        for i in brushsettings.inputs:
            w = ByInputWidget(self.app, i, setting)
            self.byinputwidgets.append(w)
            vbox.pack_start(w, expand=False)
            vbox.pack_start(gtk.HSeparator(), expand=False)

        self.set_size_request(450, 500)

    def set_fixed_value_clicked_cb(self, widget, adj, value):
        adj.set_value(value);

    def brush_selected_cb(self, brush_selected):
        # update curves
        for w in self.byinputwidgets:
            w.reread()

class ByInputWidget(gtk.VBox):
    "the gui elements that change the response to one input"
    def __init__(self, app, input, setting):
        gtk.VBox.__init__(self)
        self.tooltips = gtk.Tooltips()

        self.set_spacing(5)

        self.app = app
        self.input = input
        self.setting = setting

        self.scale_x_adj = gtk.Adjustment(value=1.0, lower=0.01, upper=20.0, step_incr=0.01, page_incr=0.1)
        diff = setting.max - setting.min
        self.scale_y_adj = gtk.Adjustment(value=diff/4.0, lower=-diff, upper=+diff, step_incr=0.01, page_incr=0.1)

        self.scale_x_adj.connect('value-changed', self.user_changes_cb)
        self.scale_y_adj.connect('value-changed', self.user_changes_cb)

        eb = gtk.EventBox()
        l = gtk.Label()
        l.set_markup('By <b>%s</b>' % input.name)
        l.set_alignment(0.0, 0.0)
        self.tooltips.set_tip(eb, input.tooltip)
        eb.add(l)
        self.pack_start(eb, expand=False)
        
        hbox = gtk.HBox()
        self.pack_start(hbox, expand=False)
        #hbox.pack_start(gtk.Label('by ' + input.name), expand=False)
        scale = self.base_value_hscale = gtk.HScale(self.scale_y_adj)
        scale.set_digits(2)
        scale.set_draw_value(True)
        scale.set_value_pos(gtk.POS_LEFT)
        hbox.pack_start(scale, expand=True)
        b = gtk.Button("0.0")
        b.connect('clicked', self.set_fixed_value_clicked_cb, self.scale_y_adj, 0.0)
        hbox.pack_start(b, expand=False)

        t = gtk.Table(4, 4)
        c = self.curve_widget = CurveWidget(self.user_changes_cb)
        t.attach(c, 0, 3, 0, 3, gtk.EXPAND | gtk.FILL, gtk.EXPAND | gtk.FILL, 5, 0)
        l1 = gtk.SpinButton(self.scale_y_adj); l1.set_digits(2)
        l2 = gtk.Label(' 0.0')
        l3 = gtk.Label('') # actually -l1; FIXME: maybe implement this later
        t.attach(l1, 3, 4, 0, 1, 0, 0, 5, 0)
        t.attach(l2, 3, 4, 1, 2, 0, gtk.EXPAND, 5, 0)
        t.attach(l3, 3, 4, 2, 3, 0, 0, 5, 0)
        l4 = gtk.Label('0.0')
        l5 = gtk.Label('')
        l6 = gtk.SpinButton(self.scale_x_adj); l6.set_digits(2)
        t.attach(l4, 0, 1, 3, 4, 0, 0, 5, 0)
        t.attach(l5, 1, 2, 3, 4, gtk.EXPAND, 0, 5, 0)
        t.attach(l6, 2, 3, 3, 4, 0, 0, 5, 0)

        expander = self.expander = gtk.Expander(label='Details')
        expander.add(t)
        expander.set_expanded(False)

        self.pack_start(expander, expand=False)

        self.reread()

    def set_fixed_value_clicked_cb(self, widget, adj, value):
        adj.set_value(value);

    def user_changes_cb(self, widget):
        scale_x, scale_y = self.scale_x_adj.get_value(), self.scale_y_adj.get_value()
        curve_points = self.curve_widget.points[1:]
        if scale_y:
            allzero = True
            brush_points = []
            for p in curve_points:
                x, y = p
                x = x * scale_x
                y = (0.5-y) * 2.0 * scale_y
                brush_points += [x, y]
                if y != 0:
                    allzero = False
            while len(brush_points) < 8:
                brush_points += [0.0, 0.0]
            if allzero:
                brush_points = None
        else:
            brush_points = None
        self.app.brush.settings[self.setting.index].set_points(self.input, brush_points)

    def reconsider_details(self):
        s = self.app.brush.settings[self.setting.index]
        if s.has_input_nonlinear(self.input):
            self.expander.set_expanded(True)
        else:
            self.expander.set_expanded(False)

    def reread(self):
        brush_points = self.app.brush.settings[self.setting.index].points[self.input.index]
        curve_points = []

        scale_y = 0.0
        scale_x = 1.0
        if brush_points is None:
            self.curve_widget.points = [(0.0, 0.5), (1.0, 0.0)]
        else:
            for i in range(4):
                x, y = brush_points[2*i], brush_points[2*i+1]
                if x == 0.0:
                    break
                if scale_y < abs(y):
                    scale_y = abs(y)
                curve_points.append((x, y))
            scale_x = curve_points[-1][0]
            assert scale_x > 0
            assert scale_y > 0
            if curve_points[0][1] < 0:
                scale_y = - scale_y # make the first line always go upwards

            final_curve_points = [(0.0, 0.5)]
            for p in curve_points:
                x, y = p
                x = x / scale_x
                y = -y / 2.0 / scale_y + 0.5
                final_curve_points.append((x, y))

            self.curve_widget.points = final_curve_points
        self.curve_widget.queue_draw()

        #print 'scale:', scale_x, scale_y
        self.scale_x_adj.set_value(scale_x)
        self.scale_y_adj.set_value(scale_y)

        self.reconsider_details()

RADIUS = 4
class CurveWidget(gtk.DrawingArea):
    "modify a (restricted) nonlinear curve"
    def __init__(self, changed_cb):
        gtk.DrawingArea.__init__(self)
        self.points = [(0.0, 0.5), (1.0, 0.0)]
        self.maxpoints = 5
        self.grabbed = None
        self.changed_cb = changed_cb

        self.connect("expose-event", self.expose_cb)
        self.connect("button-press-event", self.button_press_cb)
        self.connect("button-release-event", self.button_release_cb)
        self.connect("motion-notify-event", self.motion_notify_cb)
	self.set_events(gtk.gdk.EXPOSURE_MASK |
                        gtk.gdk.BUTTON_PRESS_MASK |
                        gtk.gdk.BUTTON_RELEASE_MASK |
                        gtk.gdk.POINTER_MOTION_MASK
                        )
        self.set_size_request(300, 200)

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
            #assert self.grabbed is None # This did happen. I think it's save to ignore? 
            # I guess it's because gtk can generate button press event without corresponding release.
            self.grabbed = nearest

    def button_release_cb(self, widget, event):
        if not event.button == 1: return
        if self.grabbed:
            i = self.grabbed
            if self.points[i] is None:
                self.points.pop(i)
        self.grabbed = None
        # notify user of the widget
        self.changed_cb(self)
            
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
