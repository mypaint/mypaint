"the main drawing window"
import gtk
import mydrawwidget


class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app

        self.set_title('MyPaint')
        def delete_event_cb(window, event, app): app.quit()
        self.connect('delete-event', delete_event_cb, self.app)
        self.set_size_request(600, 400)
        vbox = gtk.VBox()
        self.add(vbox)

        self.create_ui()
        vbox.pack_start(self.ui.get_widget('/Menubar'), expand=False)

        # maximum useful size
        # FIXME: that's /my/ screen resolution
        self.mdw = mydrawwidget.MyDrawWidget(1280, 1024);
        self.mdw.clear()
        self.mdw.set_brush(self.app.brush)
        vbox.pack_start(self.mdw)

        self.statusbar = sb = gtk.Statusbar()
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
            <menu action='ContextMenu'>
              <menuitem action='Context00'/>
              <menuitem action='Context01'/>
              <menuitem action='Context02'/>
              <menuitem action='Context03'/>
              <menuitem action='Context04'/>
              <menuitem action='Context05'/>
              <menuitem action='Context06'/>
              <menuitem action='Context07'/>
              <menuitem action='Context08'/>
              <menuitem action='Context09'/>
              <menuitem action='ContextHelp'/>
            </menu>
          </menubar>
        </ui>"""
        actions = [
            ('FileMenu',    None, 'File'),
            ('Clear',       None, 'Clear', '3', 'blank everything', self.clear_cb),
            ('NewWindow',   None, 'New Window', '<control>N', None, self.new_window_cb),
            ('Open',        None, 'Open', '<control>O', None, self.open_cb),
            ('Save',        None, 'Save', '<control>S', None, self.save_cb),
            ('Quit',        None, 'Quit', '<control>Q', None, self.quit_cb),
            ('BrushMenu',   None, 'Brush'),
            ('InvertColor', None, 'Invert Color', None, None, self.invert_color_cb),
            ('ContextMenu', None, 'Context'),
            ('Context00',    None, 'Context 0', 'a', None, self.context_cb),
            ('Context01',    None, 'Context 1', 's', None, self.context_cb),
            ('Context02',    None, 'Context 2', 'd', None, self.context_cb),
            ('Context03',    None, 'Context 3', 'f', None, self.context_cb),
            ('Context04',    None, 'Context 4', None, None, self.context_cb),
            ('Context05',    None, 'Context 5', None, None, self.context_cb),
            ('Context06',    None, 'Context 6', None, None, self.context_cb),
            ('Context07',    None, 'Context 7', None, None, self.context_cb),
            ('Context08',    None, 'Context 8', None, None, self.context_cb),
            ('Context09',    None, 'Context 9', None, None, self.context_cb),
            ('ContextHelp', None, 'How to use this?', None, None, self.context_help_cb),
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
        self.app.brush.invert_color()
        
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

            pixbuf = gtk.gdk.pixbuf_new_from_file(filename)
            self.mdw.set_from_pixbuf (pixbuf)
            self.statusbar.push(1, 'Loaded from' + filename)

        dialog.destroy()
        
    def save_cb(self, action):
        dialog = gtk.FileChooserDialog("Save..", self,
                                       gtk.FILE_CHOOSER_ACTION_SAVE,
                                       (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                        gtk.STOCK_SAVE, gtk.RESPONSE_OK))
        dialog.set_default_response(gtk.RESPONSE_OK)

        filter = gtk.FileFilter()
        filter.set_name("png")
        filter.add_pattern("*.png")
        dialog.add_filter(filter)

        dialog.hide()

        if dialog.run() == gtk.RESPONSE_OK:
            filename = dialog.get_filename()

            pixbuf = self.mdw.get_as_pixbuf()
            pixbuf.save(filename, 'png')
            self.statusbar.push(1, 'Saved to' + filename)

        dialog.destroy()

    def quit_cb(self, action):
        self.app.quit()

    def context_cb(self, action):
        # TODO: this context-thing is not very useful like that, is it?
        #       You overwrite your settings too easy by accident.
        # - seperate set/restore commands?
        # - not storing settings under certain circumstances?
        # - think about other stuff... brush history, only those actually used, etc...
        i = int(action.get_name()[-2:])
        context = self.app.contexts[i]
        if self.app.selected_context is not None:
            if context is not self.app.selected_context:
                # save current settings to old context
                b = self.app.selected_context
                b.copy_settings_from(self.app.brush)
                # OPTIMIZE: this scales the bitmap down as thumbnail, which will never be used
                preview = self.app.brushselection_window.get_preview_pixbuf()
                b.update_preview(preview)
                b.name = 'context%02d' % i
                b.save(self.app.brushpath) # OPTIMIZE: later?
        self.app.selected_context = context
        self.app.select_brush(context)
        self.app.brushselection_window.set_preview_pixbuf(context.preview)

    def context_help_cb(self, action):
        print "TODO"
