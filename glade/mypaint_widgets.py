# Support module for Glade. Just imports all the widgets declared in
# mypaint_widgets.xml into a single namespace so that glade_python_init can
# do its thing.

from gui.tileddrawwidget import TiledDrawWidget
from gui.pixbuflist import PixbufList
from gui.elastic import ElasticWindow, ElasticVBox, ElasticExpander
from gui.curve import CurveWidget
