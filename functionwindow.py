# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY. See the COPYING file for more details.

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

        self.block_user_changes_cb = True

        lower = -20.0
        upper = +20.0
        if input.hard_min is not None: lower = input.hard_min
        if input.hard_max is not None: upper = input.hard_max
        self.xmin_adj = gtk.Adjustment(value=input.soft_min, lower=lower, upper=upper-0.1, step_incr=0.01, page_incr=0.1)
        self.xmax_adj = gtk.Adjustment(value=input.soft_max, lower=lower+0.1, upper=upper, step_incr=0.01, page_incr=0.1)

        diff = setting.max - setting.min
        self.scale_y_adj = gtk.Adjustment(value=diff/4.0, lower=-diff, upper=+diff, step_incr=0.01, page_incr=0.1)

        self.xmin_adj.connect('value-changed', self.user_changes_cb)
        self.xmax_adj.connect('value-changed', self.user_changes_cb)
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
        l3 = gtk.Label('%.2f' % -self.scale_y_adj.get_value())
        self.negative_scale_label = l3 # will have to update this one
        t.attach(l1, 3, 4, 0, 1, 0, 0, 5, 0)
        t.attach(l2, 3, 4, 1, 2, 0, gtk.EXPAND, 5, 0)
        t.attach(l3, 3, 4, 2, 3, 0, 0, 5, 0)
        l4 = gtk.SpinButton(self.xmin_adj); l4.set_digits(2)
        l5 = gtk.Label('')
        l6 = gtk.SpinButton(self.xmax_adj); l6.set_digits(2)
        t.attach(l4, 0, 1, 3, 4, 0, 0, 5, 0)
        t.attach(l5, 1, 2, 3, 4, gtk.EXPAND, 0, 5, 0)
        t.attach(l6, 2, 3, 3, 4, 0, 0, 5, 0)

        expander = self.expander = gtk.Expander(label='Details')
        expander.add(t)
        expander.set_expanded(False)

        self.pack_start(expander, expand=False)

        self.reread()

    def set_fixed_value_clicked_cb(self, widget, adj, value):
        adj.set_value(value)

    def user_changes_cb(self, widget):
        if self.block_user_changes_cb:
            return

        # 1. verify and constrain the changes
        scale_y = self.scale_y_adj.get_value()
        xmax = self.xmax_adj.get_value()
        xmin = self.xmin_adj.get_value()
        if xmax <= xmin:
            # change the other one
            if widget is self.xmax_adj:
                self.xmin_adj.set_value(xmax - 0.1)
            elif widget is self.xmin_adj:
                self.xmax_adj.set_value(xmin + 0.1)
            else:
                assert False
            return # this function caused itself to be called

        assert xmax > xmin

        # 2. update the brush
        if scale_y:
            allzero = True
            brush_points = [self.point_widget2real(p) for p in self.curve_widget.points]
            nonzero = [True for x, y in brush_points if y != 0]
            if not nonzero:
                brush_points = []
        else:
            brush_points = []
        self.app.brush.settings[self.setting.index].set_points(self.input, brush_points)

        # 3. update display
        self.negative_scale_label.set_text('%.2f' % -scale_y)
        self.update_graypoint()

    def update_graypoint(self):
        # highlight (x_normal, 0)
        # the user should make the curve go through this point
        self.curve_widget.graypoint = (self.get_x_normal(), 0.5)
        self.curve_widget.queue_draw()

    def reconsider_details(self):
        s = self.app.brush.settings[self.setting.index]
        if s.has_input_nonlinear(self.input):
            self.expander.set_expanded(True)
        else:
            self.expander.set_expanded(False)

    def point_real2widget(self, (x, y)):
        scale_y = self.scale_y_adj.get_value()
        xmax = self.xmax_adj.get_value()
        xmin = self.xmin_adj.get_value()
        scale_x = xmax - xmin

        assert scale_x
        if scale_y == 0:
            y = None
        else:
            y = -(y/scale_y/2.0)+0.5 
        x = (x-xmin)/scale_x
        return (x, y)

    def point_widget2real(self, (x, y)):
        scale_y = self.scale_y_adj.get_value()
        xmax = self.xmax_adj.get_value()
        xmin = self.xmin_adj.get_value()
        scale_x = xmax - xmin

        x = xmin + (x * scale_x)
        y = (0.5-y) * 2.0 * scale_y
        return (x, y)

    def get_x_normal(self):
        # return widget x coordinate of the "normal" input value
        return self.point_real2widget((self.input.normal, 0.0))[0]

    def reread(self):

        brush_points = self.app.brush.settings[self.setting.index].points[self.input.index]

        brush_points_zero = [(self.input.soft_min, 0.0), (self.input.soft_max, 0.0)]
        if not brush_points:
            brush_points = brush_points_zero

        xmin, xmax = brush_points[0][0], brush_points[-1][0]
        assert xmax > xmin
        assert max([x for x, y in brush_points]) == xmax
        assert min([x for x, y in brush_points]) == xmin

        y_min = min([y for x, y in brush_points])
        y_max = max([y for x, y in brush_points])
        scale_y = max(abs(y_min), abs(y_max))
        scale_x = xmax - xmin

        # choose between scale_y and -scale_y (arbitrary)
        if brush_points[0][1] > brush_points[-1][1]:
            scale_y = -scale_y

        if not scale_y:
            brush_points = brush_points_zero
            # if xmin/xmax were non-default, reset them
            xmin = self.input.soft_min
            xmax = self.input.soft_max
            scale_x = xmax - xmin

        # xmin, xmax, scale_x, scale_y are fixed now
        self.block_user_changes_cb = True
        self.xmax_adj.set_value(xmax)
        self.xmin_adj.set_value(xmin)
        self.scale_y_adj.set_value(scale_y)
        self.block_user_changes_cb = False

        curve_points = [self.point_real2widget(p) for p in brush_points]

        if not scale_y:
            # note, curve_points has undefined y values (None)
            y0 = -1.0
            y1 = +1.0
            x_normal = self.get_x_normal()

            # the default line should go through zero at x_normal
            # change one of the border points to achieve this
            if x_normal >= 0.0 and x_normal <= 1.0:
                if x_normal < 0.5:
                    y0 = -0.5/(x_normal-1.0)
                    y1 = 0.0
                else:
                    y0 = 1.0
                    y1 = -0.5/x_normal + 1.0

            (x0, trash0), (x1, trash1) = curve_points
            curve_points = [(x0, y0), (x1, y1)]

        self.curve_widget.points = curve_points
        self.curve_widget.queue_draw()

        self.update_graypoint()
        self.reconsider_details()

RADIUS = 4
class CurveWidget(gtk.DrawingArea):
    "modify a (restricted) nonlinear curve"
    def __init__(self, changed_cb, magnetic=True):
        gtk.DrawingArea.__init__(self)
        self.points = [(0.0, 0.5), (1.0, 0.0)] # doesn't matter
        self.maxpoints = 8
        self.grabbed = None
        self.changed_cb = changed_cb
        self.magnetic = magnetic

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

        self.graypoint = None

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
                if y > 1.0: y = 1.0
                if y < 0.0: y = 0.0
                self.points.insert(insertpos, (x, y))
                nearest = insertpos
                self.queue_draw()

        #if nearest == 0: 
        #    # first point cannot be grabbed
        #    display = gtk.gdk.display_get_default()
        #    display.beep()
        #else:
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
        out = False # by default, the point cannot be removed by drawing it out
        if i == len(self.points)-1:
            x = 1.0 # right point stays right
        elif i == 0:
            x = 0.0 # left point stays left
        else:
            if y > 1.1 or y < -0.1: out = True
            leftbound  = self.points[i-1][0]
            rightbound = self.points[i+1][0]
            if x <= leftbound or x >= rightbound: out = True
        if out:
            self.points[i] = None
        else:
            if y > 1.0: y = 1.0
            if y < 0.0: y = 0.0
            if self.magnetic:
                if y > 0.48 and y < 0.52: y = 0.5
                if x > 0.48 and x < 0.52: x = 0.5
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

        if self.graypoint:
            x1, y1 = self.graypoint
            x1 = int(x1*width) + RADIUS
            y1 = int(y1*height) + RADIUS
            window.draw_rectangle(dark, True, x1-RADIUS-1, y1-RADIUS-1, 2*RADIUS+1, 2*RADIUS+1)

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
