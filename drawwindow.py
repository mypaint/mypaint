"the main drawing window"
import gtk
import lowlevel


class Window(gtk.Window):
    def __init__(self):
        gtk.Window.__init__(self)
        self.set_title('MyPaint')
        self.connect('delete-event', self.delete_event_cb)
        self.set_size_request(600, 400)
        vbox = gtk.VBox()
        self.add(vbox)

        self.create_ui()
        vbox.pack_start(self.ui.get_widget('/Menubar'), expand=False)

        self.mdw = mdw = lowlevel.DrawWidget()
        self.brush = lowlevel.Brush()
        mdw.set_brush(self.brush)
        vbox.pack_start(mdw)

        self.staturbar = sb = gtk.Statusbar()
        vbox.pack_end(sb, expand=False)
        sb.push(0, "hello world")
        
    def create_ui(self):
        ag = gtk.ActionGroup('WindowActions')
        ui_string = """<ui>
          <menubar name='Menubar'>
            <menu action='FileMenu'>
              <menuitem action='Clear'/>
              <menuitem action='NewWindow'/>
              <menuitem action='Open'/>
              <menuitem action='Save'/>
              <menuitem action='Quit'/>
            </menu>
            <menu action='BrushMenu'>
              <menuitem action='InvertColor'/>
            </menu>
          </menubar>
        </ui>"""
        actions = [
            ('FileMenu', None, 'File'),
            ('Clear',    None, 'Clear', '3', 'blank everything', self.clear_cb),
            ('NewWindow',None, 'New Window', '<control>N', None, self.new_window_cb),
            ('Open',     gtk.STOCK_OPEN, 'Open', '<control>O', None, self.open_cb),
            ('Save',     gtk.STOCK_SAVE, 'Save', '<control>S', None, self.save_cb),
            ('Quit',     gtk.STOCK_QUIT, 'Quit', None, None, self.quit_cb),
            ('BrushMenu', None, 'Brush'),
            ('InvertColor', None, 'Invert Color', None, None, self.invert_color_cb),
            ]
        ag.add_actions(actions)
        self.ui = gtk.UIManager()
        self.ui.insert_action_group(ag, 0)
        self.ui.add_ui_from_string(ui_string)
        self.add_accel_group(self.ui.get_accel_group())

    def new_window_cb(self, action):
        # FIXME: is it really done like that?
        w = Window()
        w.show_all()
        #gtk.main()

    def clear_cb(self, action):
        self.mdw.clear()
        
    def invert_color_cb(self, action):
        self.brush.invert_color()
        
    def open_cb(self, action):
        dialog = gtk.FileChooserDialog("Open..", self,
                                       gtk.FILE_CHOOSER_ACTION_OPEN,
                                       (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                        gtk.STOCK_OPEN, gtk.RESPONSE_OK))
        dialog.set_default_response(gtk.RESPONSE_OK)

        filter = gtk.FileFilter()
        filter.set_name("png")
        filter.add_pattern("*.png")
        dialog.add_filter(filter)

        dialog.hide()

        if dialog.run() == gtk.RESPONSE_OK:
            filename = dialog.get_filename()
            self.statusbar.push(1, 'TODO: now open ' + filename)

        dialog.destroy()
        
    def save_cb(self, action):
        self.statusbar.push(1, 'TODO: save dialog')

    def close_cb(self, action):
        self.hide()
        gtk.main_quit()

    def quit_cb(self, action):
        raise SystemExit

    def delete_event_cb(self, window, event):
        gtk.main_quit()

