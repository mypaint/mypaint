"tune brush window"
import gtk
import brush


class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app
        self.app.brush_selected_callbacks.append(self.brush_selected_cb)

        self.set_title('Brush settings')

        scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.POLICY_NEVER, gtk.POLICY_AUTOMATIC)
        self.add(scroll)

        # FIXME: why does the scrolledwindow not work any more?

        table = gtk.Table(3, len(brush.brushsettings))
        #table.set_border_width(4)
        #table.set_col_spacings(15)
        scroll.add_with_viewport(table)

        self.tooltips = gtk.Tooltips()

        # common callbacks
        def default_clicked_cb(window, adj, default):
            adj.set_value(default)
        def value_changed_cb(adj, index, app):
            app.brush.set_setting(index, adj.get_value())
        self.adj = []
        self.app.brush_adjustment = {}
        for s in brush.brushsettings:
            eb = gtk.EventBox()
            l = gtk.Label(s.name)
            l.set_alignment(0, 0.5)
            self.tooltips.set_tip(eb, s.tooltip)
            eb.add(l)

            adj = gtk.Adjustment(value=s.default, lower=s.min, upper=s.max, step_incr=0.01, page_incr=0.1)
            adj.connect('value-changed', value_changed_cb, s.index, self.app)
            self.adj.append(adj)
            self.app.brush_adjustment[s.cname] = adj
            h = gtk.HScale(adj)
            h.set_digits(2)
            h.set_draw_value(True)
            h.set_value_pos(gtk.POS_LEFT)

            #sb = gtk.SpinButton(adj, climb_rate=0.1, digits=2)
            b = gtk.Button("%.1f" % s.default)
            b.connect('clicked', default_clicked_cb, adj, s.default)

            table.attach(eb, 0, 1, s.index, s.index+1, gtk.FILL, gtk.FILL, 5, 0)
            table.attach(h, 1, 2, s.index, s.index+1, gtk.EXPAND | gtk.FILL, gtk.EXPAND | gtk.FILL)
            table.attach(b, 2, 3, s.index, s.index+1, gtk.FILL, gtk.FILL)

        self.set_size_request(450, 500)

    def brush_selected_cb(self, brush_selected):
        for s in brush.brushsettings:
            self.adj[s.index].set_value(self.app.brush.get_setting(s.index))



