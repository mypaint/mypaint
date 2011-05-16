# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""
Widgets to manipulate the input dependency of a single brush setting.
"""

from gettext import gettext as _
import gtk
from brushlib import brushsettings
import windowing

class BrushInputsWidget(gtk.VBox):
    def __init__(self, app):
        gtk.VBox.__init__(self)

        self.app = app
        self.byinputwidgets = {}
        self.setting = None
        self.init_ui()

    def init_ui(self):
        scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.POLICY_NEVER, gtk.POLICY_ALWAYS)
        self.pack_start(scroll)

        vbox = gtk.VBox()
        scroll.add_with_viewport(vbox)

        l = gtk.Label()
        l.set_markup(_('Base Value'))
        l.set_alignment(0.0, 0.0)
        l.xpad = 5
        vbox.pack_start(l, expand=False)

        hbox = gtk.HBox()
        vbox.pack_start(hbox, expand=False)
        scale = self.base_value_hscale = gtk.HScale()
        scale.set_digits(2)
        scale.set_draw_value(True)
        scale.set_value_pos(gtk.POS_LEFT)
        hbox.pack_start(scale, expand=True)
        b = self.default_value_button = gtk.Button()
        b.connect('clicked', self.set_default_value_cb)
        hbox.pack_start(b, expand=False)

        vbox.pack_start(gtk.HSeparator(), expand=False)

        for i in brushsettings.inputs:
            w = ByInputWidget(self.app, i)
            self.byinputwidgets[i.name] = w
            vbox.pack_start(w, expand=False)
            vbox.pack_start(gtk.HSeparator(), expand=False)

    def set_brushsetting(self, setting, adj):
        self.setting = setting
        self.base_value_hscale.set_adjustment(adj)

        b = self.default_value_button
        b.set_label("%.1f" % setting.default)

        for widget in self.byinputwidgets.itervalues():
            widget.set_brushsetting(setting)

    def set_default_value_cb(self, widget):
        self.app.brush.set_base_value(self.setting.cname, self.setting.default)


def points_equal(points_a, points_b):
    if len(points_a) != len(points_b):
        return False
    for a, b in zip(points_a, points_b):
        for v1, v2 in zip(a, b):
            if abs(v1-v2) > 0.0001:
                return False
    return True

class ByInputWidget(gtk.VBox):
    "the gui elements that change the response to one input"
    def __init__(self, app, input):
        gtk.VBox.__init__(self)

        self.set_spacing(5)

        self.app = app
        self.input = input
        self.setting = None

        self.app.brush.observers.append(self.brush_modified_cb)
        self.app.brushmanager.selected_brush_observers.append(self.brush_selected_cb)

        self.points = None
        self.disable_scale_changes_cb = True

        lower = -20.0
        upper = +20.0
        if input.hard_min is not None: lower = input.hard_min
        if input.hard_max is not None: upper = input.hard_max
        self.xmin_adj = gtk.Adjustment(value=input.soft_min, lower=lower, upper=upper-0.1, step_incr=0.01, page_incr=0.1)
        self.xmax_adj = gtk.Adjustment(value=input.soft_max, lower=lower+0.1, upper=upper, step_incr=0.01, page_incr=0.1)

        self.scale_y_adj = gtk.Adjustment(value=1.0/4.0, lower=-1.0, upper=1.0, step_incr=0.01, page_incr=0.1)

        self.xmin_adj.connect('value-changed', self.scale_changes_cb)
        self.xmax_adj.connect('value-changed', self.scale_changes_cb)
        self.scale_y_adj.connect('value-changed', self.scale_changes_cb)

        l = gtk.Label()
        l.set_markup(input.dname)
        l.set_alignment(0.0, 0.0)
        l.set_tooltip_text(input.tooltip)
        self.pack_start(l, expand=False)
        
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
        c = self.curve_widget = CurveWidget(self.curvewidget_points_modified_cb)
        t.attach(c, 0, 3, 0, 3, gtk.EXPAND | gtk.FILL, gtk.EXPAND | gtk.FILL, 5, 0)
        l1 = gtk.SpinButton(self.scale_y_adj); l1.set_digits(2)
        l2 = gtk.Label('+0.0')
        l3 = gtk.Label()
        def update_negative_scale(*trash):
            l3.set_text('%+.2f' % -self.scale_y_adj.get_value())
        self.scale_y_adj.connect('value-changed', update_negative_scale)
        update_negative_scale()

        t.attach(l1, 3, 4, 0, 1, 0, 0, 5, 0)
        t.attach(l2, 3, 4, 1, 2, 0, gtk.EXPAND, 5, 0)
        t.attach(l3, 3, 4, 2, 3, 0, 0, 5, 0)
        l4 = gtk.SpinButton(self.xmin_adj); l4.set_digits(2)
        l5 = gtk.Label('')
        l6 = gtk.SpinButton(self.xmax_adj); l6.set_digits(2)
        t.attach(l4, 0, 1, 3, 4, 0, 0, 5, 0)
        t.attach(l5, 1, 2, 3, 4, gtk.EXPAND, 0, 5, 0)
        t.attach(l6, 2, 3, 3, 4, 0, 0, 5, 0)

        expander = self.expander = gtk.Expander(label=_('Details'))
        expander.add(t)
        expander.set_expanded(False)

        self.pack_start(expander, expand=False)

        self.disable_scale_changes_cb = False

    def set_brushsetting(self, setting):
        self.setting = setting
        self.reset_gui()

    def brush_selected_cb(self, managed_brush):
        if not self.setting:
            return
        self.reset_gui()

    def set_fixed_value_clicked_cb(self, widget, adj, value):
        adj.set_value(value)

    def scale_changes_cb(self, widget):
        if self.disable_scale_changes_cb:
            return
        # MVC: this is part of the controller

        # 1. verify and constrain the adjustment changes
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
            return # the adjustment change causes another call of this function

        assert xmax > xmin

        # 2. interpret the points displayed in the curvewidget
        #    according to the new scale (update the brush)
        self.curvewidget_points_modified_cb(None)
        
    def get_brushpoints_from_curvewidget(self):
        scale_y = self.scale_y_adj.get_value()
        if not scale_y:
            return []
        brush_points = [self.point_widget2real(p) for p in self.curve_widget.points]
        nonzero = [True for x, y in brush_points if y != 0]
        if not nonzero:
            return []
        return brush_points

    def curvewidget_points_modified_cb(self, widget):
        # MVC: this is part of the controller
        # update the brush with the points from curve_widget
        points = self.get_brushpoints_from_curvewidget()
        self.app.brush.set_points(self.setting.cname, self.input.name, points)

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

    def reset_gui(self):
        "update scales (adjustments) to fit the brush curve into view"
        # MVC: this is part of the view
        brush_points = self.app.brush.get_points(self.setting.cname, self.input.name)

        brush_points_zero = [(self.input.soft_min, 0.0), (self.input.soft_max, 0.0)]
        if not brush_points:
            brush_points = brush_points_zero

        # 1. update the scale

        xmin, xmax = brush_points[0][0], brush_points[-1][0]
        assert xmax > xmin
        assert max([x for x, y in brush_points]) == xmax
        assert min([x for x, y in brush_points]) == xmin

        y_min = min([y for x, y in brush_points])
        y_max = max([y for x, y in brush_points])
        scale_y = max(abs(y_min), abs(y_max))

        # choose between scale_y and -scale_y (arbitrary)
        if brush_points[0][1] > brush_points[-1][1]:
            scale_y = -scale_y

        if not scale_y:
            brush_points = brush_points_zero
            # if xmin/xmax were non-default, reset them
            xmin = self.input.soft_min
            xmax = self.input.soft_max

        self.disable_scale_changes_cb = True
        # set adjustment limits imposed by brush setting
        diff = self.setting.max - self.setting.min
        self.scale_y_adj.set_upper(+diff)
        self.scale_y_adj.set_lower(-diff)
        self.scale_y_adj.set_value(scale_y)
        # set adjustment values such that all points are visible
        self.xmax_adj.set_value(xmax)
        self.xmin_adj.set_value(xmin)
        self.disable_scale_changes_cb = False


        # 2. calculate the default curve (the one we display if there is no curve)

        curve_points_zero = [self.point_real2widget(p) for p in brush_points_zero]

        # widget x coordinate of the "normal" input value
        x_normal = self.get_x_normal()

        y0 = -1.0
        y1 = +1.0
        # the default line should go through zero at x_normal
        # change one of the border points to achieve this
        if x_normal >= 0.0 and x_normal <= 1.0:
            if x_normal < 0.5:
                y0 = -0.5/(x_normal-1.0)
                y1 = 0.0
            else:
                y0 = 1.0
                y1 = -0.5/x_normal + 1.0

        (x0, trash0), (x1, trash1) = curve_points_zero
        curve_points_zero = [(x0, y0), (x1, y1)]

        # 3. display the curve

        if scale_y:
            curve_points = [self.point_real2widget(p) for p in brush_points]
        else:
            curve_points = curve_points_zero

        assert len(curve_points) >= 2
        self.curve_widget.points = curve_points
        self.curve_widget.queue_draw()
        self.update_graypoint()

        # 4. reset the expander

        interesting = not points_equal(curve_points, curve_points_zero)
        self.expander.set_expanded(interesting)

    def get_x_normal(self):
        "returns the widget x coordinate of the 'normal' value of the input"
        return self.point_real2widget((self.input.normal, 0.0))[0]

    def update_graypoint(self):
        self.curve_widget.graypoint = (self.get_x_normal(), 0.5)
        self.curve_widget.queue_draw()

    def brush_modified_cb(self, settings):
        "update gui when the brush has been modified (by this widget or externally)"
        if not self.setting:
            return
        if self.setting.cname not in settings:
            return

        # check whether we really need to update anything
        points_old = self.get_brushpoints_from_curvewidget()
        points_new = self.app.brush.get_points(self.setting.cname, self.input.name)

        # try not to destroy changes that the user made to the widget
        # (not all gui states are stored inside the brush)
        if not points_equal(points_old, points_new):
            self.reset_gui()

        self.update_graypoint()


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
            dist = abs(px - x) + 0.5*abs(py - y)
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
            # last point stays right
            leftbound = rightbound = 1.0
        elif i == 0:
            # first point stays left
            leftbound = rightbound = 0.0
        else:
            # other points can be dragged out
            if y > 1.1 or y < -0.1: out = True
            leftbound  = self.points[i-1][0]
            rightbound = self.points[i+1][0]
            if x <= leftbound - 0.02 or x >= rightbound + 0.02: out = True
        if out:
            self.points[i] = None
        else:
            if y > 1.0: y = 1.0
            if y < 0.0: y = 0.0
            if self.magnetic:
                if y > 0.48 and y < 0.52: y = 0.5
                if x > 0.48 and x < 0.52: x = 0.5
            if x < leftbound: x = leftbound
            if x > rightbound: x = rightbound
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
