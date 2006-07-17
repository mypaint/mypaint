import gtk, os
import drawwindow, brushsettingswindow, brushselectionwindow, colorselectionwindow
import brush

class Application: # singleton
    def __init__(self, prefix, confpath, loadimage):
        self.confpath = confpath

        datapaths = ['./', '/share/mypaint/', '/usr/local/share/mypaint/']
        if prefix:
            datapaths.append(prefix + 'share/mypaint/')
        self.datapath = None
        for p in datapaths:
            if os.path.isdir(p + 'brushes/'):
                self.datapath = p
                break
        if not self.datapath:
            print 'Default brush collection not found! Searched:'
            print ' '.join(datapaths)
            raise SystemExit

        self.user_brushpath = self.confpath + 'brushes/'
        self.stock_brushpath = self.datapath + 'brushes/'

        if not os.path.isdir(self.confpath):
            os.mkdir(self.confpath)
            print 'Created', self.confpath
        if not os.path.isdir(self.user_brushpath):
            os.mkdir(self.user_brushpath)

        self.init_brushes()

        self.image_windows = []
        self.new_image_window()

        w = self.brushSettingsWindow = brushsettingswindow.Window(self)

        w = self.brushSelectionWindow = brushselectionwindow.Window(self)
        w.show_all()

        w = self.colorSelectionWindow = colorselectionwindow.Window(self)

        gtk.accel_map_load(self.confpath + 'accelmap.conf')

        if loadimage:
            self.image_windows[0].open_file(loadimage)

    def init_brushes(self):
        self.brush = brush.Brush(self)
        self.brushes = []
        self.selected_brush = None
        self.brush_selected_callbacks = [self.brush_selected_cb]
        self.contexts = []
        for i in range(10):
            c = brush.Brush(self)
            c.name = 'context%02d' % i
            self.contexts.append(c)
        self.selected_context = None

        # find all brush names to load
        deleted = []
        filename = self.user_brushpath + 'deleted.conf'
        if os.path.exists(filename): 
            for name in open(filename).readlines():
                deleted.append(name.strip())
        def listbrushes(path):
            return [filename[:-4] for filename in os.listdir(path) if filename.endswith('.myb')]
        stock_names = listbrushes(self.stock_brushpath)
        user_names =  listbrushes(self.user_brushpath)
        stock_names = [name for name in stock_names if name not in deleted and name not in user_names]
        loadnames_unsorted = user_names + stock_names
        loadnames_sorted = []

        # sort them
        for path in [self.user_brushpath, self.stock_brushpath]:
            filename = path + 'order.conf'
            if not os.path.exists(filename): continue
            for name in open(filename).readlines():
                name = name.strip()
                if name in loadnames_sorted: continue
                if name not in loadnames_unsorted: continue
                loadnames_unsorted.remove(name)
                loadnames_sorted.append(name)
        if len(loadnames_unsorted) > 3: 
            # many new brushes, do not disturb user's order
            loadnames = loadnames_sorted + loadnames_unsorted
        else:
            loadnames = loadnames_unsorted + loadnames_sorted

        for name in loadnames:
            # load brushes from disk
            b = brush.Brush(self)
            b.load(name)
            if name.startswith('context'):
                i = int(name[-2:])
                assert i >= 0 and i < 10 # 10 for now...
                self.contexts[i] = b
            else:
                self.brushes.append(b)

        if self.brushes:
            self.select_brush(self.brushes[0])

        self.brush.set_color((0, 0, 0))

    def save_brushorder(self):
        f = open(self.user_brushpath + 'order.conf', 'w')
        f.write('# this file saves brushorder\n')
        f.write('# the first one (upper left) will be selected at startup\n')
        for b in self.brushes:
            f.write(b.name + '\n')
        f.close()

    def update_statistics(self):
        # permanently update painting_time of selected brush
        if self.selected_brush:
            self.selected_brush.painting_time += self.brush.get_painting_time()
            # FIXME: save statistics elsewhere (brushes must not be saved unless modified)
            #self.selected_brush.save(self.user_brushpath)
            # just don't save the statistic for now...
        self.brush.set_painting_time(0)

    def brush_selected_cb(self, brush):
        "actually set the new brush"
        assert brush is not self.brush # self.brush never gets exchanged
        self.update_statistics()
        if brush in self.brushes:
            self.selected_brush = brush
        else:
            #print 'Warning, you have selected a brush not in the list.'
            # TODO: maybe find out parent and set this as selected_brush
            self.selected_brush = None
        if brush is not None:
            self.brush.copy_settings_from(brush)

    def select_brush(self, brush):
        for callback in self.brush_selected_callbacks:
            callback(brush)

    def hide_window_cb(self, window, event):
        # used by some of the windows
        window.hide()
        return True

    def quit(self):
        self.update_statistics()
        gtk.accel_map_save(self.confpath + 'accelmap.conf')
        d = gtk.Dialog("Really quit?",
             None,
             gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT,
             (gtk.STOCK_YES, gtk.RESPONSE_ACCEPT,
              gtk.STOCK_NO, gtk.RESPONSE_REJECT))
        if d.run() == gtk.RESPONSE_ACCEPT:
            gtk.main_quit()
            return False
        d.destroy()
        return True
        
    def new_image_window(self):
        w = drawwindow.Window(self)
        w.show_all()
        self.image_windows.append(w)
        return w


