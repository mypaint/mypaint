"tune brush window"
import gtk
import functionwindow
import brush, brushsettings

class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app
        self.app.brush_selected_callbacks.append(self.brush_selected_cb)
        self.add_accel_group(self.app.accel_group)

        self.set_title('Brush settings')
        self.connect('delete-event', self.app.hide_window_cb)

        scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.POLICY_NEVER, gtk.POLICY_AUTOMATIC)
        self.add(scroll)

        table = gtk.Table(4, len(brushsettings.settings))
        #table.set_border_width(4)
        #table.set_col_spacings(15)
        scroll.add_with_viewport(table)

        self.tooltips = gtk.Tooltips()

        self.adj = []
        self.app.brush_adjustment = {}
        for s in brushsettings.settings:
            eb = gtk.EventBox()
            l = gtk.Label(s.name)
            l.set_alignment(0, 0.5)
            self.tooltips.set_tip(eb, s.tooltip)
            eb.add(l)

            adj = gtk.Adjustment(value=s.default, lower=s.min, upper=s.max, step_incr=0.01, page_incr=0.1)
            adj.connect('value-changed', self.value_changed_cb, s.index, self.app)
            self.adj.append(adj)
            self.app.brush_adjustment[s.cname] = adj
            h = gtk.HScale(adj)
            h.set_digits(2)
            h.set_draw_value(True)
            h.set_value_pos(gtk.POS_LEFT)

            #sb = gtk.SpinButton(adj, climb_rate=0.1, digits=2)
            b = gtk.Button("%.1f" % s.default)
            b.connect('clicked', self.set_fixed_value_clicked_cb, adj, s.default)

            b2 = gtk.Button("...")
            b2.connect('clicked', self.details_clicked_cb, adj, s)

            table.attach(eb, 0, 1, s.index, s.index+1, gtk.FILL, gtk.FILL, 5, 0)
            table.attach(h, 1, 2, s.index, s.index+1, gtk.EXPAND | gtk.FILL, gtk.EXPAND | gtk.FILL)
            table.attach(b, 2, 3, s.index, s.index+1, gtk.FILL, gtk.FILL)
            table.attach(b2, 3, 4, s.index, s.index+1, gtk.FILL, gtk.FILL)

        self.functionWindows = len(brushsettings.settings) * [None]

        self.set_size_request(450, 500)

    def set_fixed_value_clicked_cb(self, widget, adj, value):
        adj.set_value(value);

    def details_clicked_cb(self, window, adj, setting):
        # FIXME: should the old window get closed automatically?
        #        Hm... probably not.
        w = self.functionWindows[setting.index]
        if w is None:
            w = functionwindow.Window(self.app, setting, adj)
            self.functionWindows[setting.index] = w
            w.show_all()
        w.hide() # maybe this helps to get it in front?
        w.show()

    def value_changed_cb(self, adj, index, app):
        app.brush.settings[index].set_base_value(adj.get_value())

    def brush_selected_cb(self, brush_selected):
        for s in brushsettings.settings:
            self.adj[s.index].set_value(self.app.brush.settings[s.index].base_value)



