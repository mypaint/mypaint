"the main drawing window"
import gtk, os
import infinitemydrawwidget

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

        self.mdw = infinitemydrawwidget.InfiniteMyDrawWidget()
        self.mdw.clear()
        self.mdw.set_brush(self.app.brush)
        vbox.pack_start(self.mdw)

        self.statusbar = sb = gtk.Statusbar()
        vbox.pack_end(sb, expand=False)
        sb.push(0, "hello world")

        
    def create_ui(self):
        ag = gtk.ActionGroup('WindowActions')
        # FIXME: this xml menu ony creates unneeded information duplication, I think.
        ui_string = """<ui>
          <menubar name='Menubar'>
            <menu action='FileMenu'>
              <menuitem action='Clear'/>
              <menuitem action='MoveLeft'/>
              <menuitem action='MoveRight'/>
              <menuitem action='MoveUp'/>
              <menuitem action='MoveDown'/>
              <separator/>
              <menuitem action='Open'/>
              <menuitem action='Save'/>
              <separator/>
              <menuitem action='Quit'/>
            </menu>
            <menu action='BrushMenu'>
              <menuitem action='InvertColor'/>
              <menuitem action='Bigger'/>
              <menuitem action='Smaller'/>
              <menuitem action='Brighter'/>
              <menuitem action='Darker'/>
            </menu>
            <menu action='ContextMenu'>
              <menuitem action='ContextStore'/>
              <separator/>
              <menuitem action='Context00'/>
              <menuitem action='Context00s'/>
              <menuitem action='Context01'/>
              <menuitem action='Context01s'/>
              <menuitem action='Context02'/>
              <menuitem action='Context02s'/>
              <menuitem action='Context03'/>
              <menuitem action='Context03s'/>
              <menuitem action='Context04'/>
              <menuitem action='Context04s'/>
              <menuitem action='Context05'/>
              <menuitem action='Context05s'/>
              <menuitem action='Context06'/>
              <menuitem action='Context06s'/>
              <menuitem action='Context07'/>
              <menuitem action='Context07s'/>
              <menuitem action='Context08'/>
              <menuitem action='Context08s'/>
              <menuitem action='Context09'/>
              <menuitem action='Context09s'/>
              <menuitem action='ContextHelp'/>
            </menu>
            <menu action='DialogMenu'>
              <menuitem action='BrushSelectionWindow'/>
              <menuitem action='BrushSettingsWindow'/>
              <menuitem action='ColorSelectionWindow'/>
            </menu>
            <menu action='DebugMenu'>
              <menuitem action='PrintInputs'/>
              <menuitem action='DontPrintInputs'/>
            </menu>
          </menubar>
        </ui>"""
        actions = [
            ('FileMenu',     None, 'File'),
            ('Clear',        None, 'Clear', '3', 'blank everything', self.clear_cb),
            ('MoveLeft',     None, 'Move left', 'h', None, self.move_cb),
            ('MoveRight',    None, 'Move right', 'l', None, self.move_cb),
            ('MoveUp',       None, 'Move up', 'k', None, self.move_cb),
            ('MoveDown',     None, 'Move down', 'j', None, self.move_cb),
            #('NewWindow',    None, 'New Window', '<control>N', None, self.new_window_cb),
            ('Open',         None, 'Open', '<control>O', None, self.open_cb),
            ('Save',         None, 'Save', '<control>S', None, self.save_cb),
            ('Quit',         None, 'Quit', '<control>Q', None, self.quit_cb),
            ('BrushMenu',    None, 'Brush'),
            ('InvertColor',  None, 'Invert Color', None, None, self.invert_color_cb),
            ('Brighter',     None, 'Brighter', None, None, self.brighter_cb),
            ('Darker',       None, 'Darker', None, None, self.darker_cb),
            ('Bigger',       None, 'Bigger', None, None, self.brush_bigger_cb),
            ('Smaller',      None, 'Smaller', None, None, self.brush_smaller_cb),
            ('ContextMenu',  None, 'Context'),
            ('Context00',    None, 'Context 0', None, None, self.context_cb),
            ('Context00s',   None, 'set Context 0', None, None, self.context_cb),
            ('Context01',    None, 'Context 1', None, None, self.context_cb),
            ('Context01s',   None, 'set Context 1', None, None, self.context_cb),
            ('Context02',    None, 'Context 2', None, None, self.context_cb),
            ('Context02s',   None, 'set Context 2', None, None, self.context_cb),
            ('Context03',    None, 'Context 3', None, None, self.context_cb),
            ('Context03s',   None, 'set Context 3', None, None, self.context_cb),
            ('Context04',    None, 'Context 4', None, None, self.context_cb),
            ('Context04s',   None, 'set Context 4', None, None, self.context_cb),
            ('Context05',    None, 'Context 5', None, None, self.context_cb),
            ('Context05s',   None, 'set Context 5', None, None, self.context_cb),
            ('Context06',    None, 'Context 6', None, None, self.context_cb),
            ('Context06s',   None, 'set Context 6', None, None, self.context_cb),
            ('Context07',    None, 'Context 7', None, None, self.context_cb),
            ('Context07s',   None, 'set Context 7', None, None, self.context_cb),
            ('Context08',    None, 'Context 8', None, None, self.context_cb),
            ('Context08s',   None, 'set Context 8', None, None, self.context_cb),
            ('Context09',    None, 'Context 9', None, None, self.context_cb),
            ('Context09s',   None, 'set Context 9', None, None, self.context_cb),
            ('ContextStore', None, 'set Context last selected', None, None, self.context_cb),
            ('ContextHelp',  None, 'How to use this?', None, None, self.context_help_cb),
            ('DialogMenu',  None, 'Dialogs'),
            ('BrushSelectionWindow',  None, 'brush list', None, None, self.toggleBrushSelectionWindow_cb),
            ('BrushSettingsWindow',   None, 'brush settings', None, None, self.toggleBrushSettingsWindow_cb),
            ('ColorSelectionWindow',  None, 'color', None, None, self.toggleColorSelectionWindow_cb),
            ('DebugMenu',  None, 'Debug'),
            ('PrintInputs', None, 'Print brush input values to stdout', None, None, self.print_inputs_cb),
            ('DontPrintInputs', None, 'Stop printing them', None, None, self.dont_print_inputs_cb),
            ]
        ag.add_actions(actions)
        self.ui = gtk.UIManager()
        self.ui.insert_action_group(ag, 0)
        self.ui.add_ui_from_string(ui_string)
        self.app.accel_group = self.ui.get_accel_group()
        self.add_accel_group(self.app.accel_group)


    def toggleWindow(self, w):
        if w.get_property('visible'):
            w.hide()
        else:
            w.show()
    def toggleBrushSelectionWindow_cb(self, action):
        self.toggleWindow(self.app.brushSelectionWindow)
    def toggleBrushSettingsWindow_cb(self, action):
        self.toggleWindow(self.app.brushSettingsWindow)
    def toggleColorSelectionWindow_cb(self, action):
        self.toggleWindow(self.app.colorSelectionWindow)

    def print_inputs_cb(self, action):
        self.app.brush.set_print_inputs(1)
    def dont_print_inputs_cb(self, action):
        self.app.brush.set_print_inputs(0)

    def new_window_cb(self, action):
        # FIXME: is it really done like that?
        #w = Window()
        #w.show_all()
        #gtk.main()
        print "Not really implemented."
        #pass

    def clear_cb(self, action):
        self.mdw.clear()
        
    def invert_color_cb(self, action):
        self.app.brush.invert_color()
        
    def brush_bigger_cb(self, action):
        adj = self.app.brush_adjustment['radius_logarithmic']
        adj.set_value(adj.get_value() + 0.3)
        
    def brush_smaller_cb(self, action):
        adj = self.app.brush_adjustment['radius_logarithmic']
        adj.set_value(adj.get_value() - 0.3)

    def brighter_cb(self, action):
        cs = self.app.colorSelectionWindow 
        cs.update()
        h, s, v = cs.get_color_hsv()
        v += 0.08
        cs.set_color_hsv((h, s, v))
        
    def darker_cb(self, action):
        cs = self.app.colorSelectionWindow 
        cs.update()
        h, s, v = cs.get_color_hsv()
        v -= 0.08
        cs.set_color_hsv((h, s, v))
        
    def open_file(self, filename):
        self.mdw.load(filename)
        self.statusbar.push(1, 'Loaded from' + filename)

    def save_file(self, filename):
        self.mdw.save(filename)
        self.statusbar.push(1, 'Saved to' + filename)

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
            self.open_file(dialog.get_filename())

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
            if os.path.exists(filename):
                d2 = gtk.Dialog("Overwrite?",
                     self,
                     gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT,
                     (gtk.STOCK_YES, gtk.RESPONSE_ACCEPT,
                      gtk.STOCK_NO, gtk.RESPONSE_REJECT))
                if d2.run() != gtk.RESPONSE_ACCEPT:
                    filename = None
                d2.destroy()
            if filename:
                self.save_file(dialog.get_filename())

        dialog.destroy()

    def quit_cb(self, action):
        self.app.quit()

    def move_cb(self, action):
        #step = 20
        step = min(self.mdw.window.get_size()) / 5
        name = action.get_name()
        if name == 'MoveLeft':
            self.mdw.scroll(-step, 0)
        elif name == 'MoveRight':
            self.mdw.scroll(+step, 0)
        elif name == 'MoveUp':
            self.mdw.scroll(0, -step)
        elif name == 'MoveDown':
            self.mdw.scroll(0, +step)
        else:
            assert 0

    def context_cb(self, action):
        # TODO: this context-thing is not very useful like that, is it?
        #       You overwrite your settings too easy by accident.
        # - seperate set/restore commands?
        # - not storing settings under certain circumstances?
        # - think about other stuff... brush history, only those actually used, etc...
        name = action.get_name()
        store = False
        if name == 'ContextStore':
            context = self.app.selected_context
            if not context:
                print 'No context was selected, ignoring store command.'
                return
            store = True
        else:
            if name.endswith('s'):
                store = True
                name = name[:-1]
            i = int(name[-2:])
            context = self.app.contexts[i]
        self.app.selected_context = context
        if store:
            context.copy_settings_from(self.app.brush)
            preview = self.app.brushSelectionWindow.get_preview_pixbuf()
            context.update_preview(preview)
            context.save(self.app.brushpath)
        else: # restore
            self.app.select_brush(context)
            self.app.brushSelectionWindow.set_preview_pixbuf(context.preview)

    def context_help_cb(self, action):
        print "TODO"
