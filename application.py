import gtk
import drawwindow, brushsettingswindow

class Application: # singleton
    def __init__(self):
        self.image_windows = []

        self.brushsettings_window = brushsettingswindow.Window(self)
        self.brushsettings_window.show_all()
        self.new_image_window()
        
    def new_image_window(self):
        w = drawwindow.Window(self)
        w.show_all()
        self.image_windows.append(w)
        return w
