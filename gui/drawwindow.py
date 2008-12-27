# coding: utf8
#
# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY. See the COPYING file for more details.

"the main drawing window"
MYPAINT_VERSION="0.6.0-svn"
import gtk, os, zlib, random, re, math
from gtk import gdk, keysyms
import tileddrawwidget, colorselectionwindow
from lib import document #, command
from time import time
from glob import glob

class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app

        self.set_title('MyPaint')
        self.connect('delete-event', self.quit_cb)
        self.connect('key-press-event', self.key_press_event_cb_before)
        self.connect('key-release-event', self.key_release_event_cb_before)
        self.connect_after('key-press-event', self.key_press_event_cb_after)
        self.connect_after('key-release-event', self.key_release_event_cb_after)
        self.connect("button-press-event", self.button_press_cb)
        self.connect("button-release-event", self.button_release_cb)
        self.connect("scroll-event", self.scroll_cb)
        self.set_default_size(600, 400)
        vbox = gtk.VBox()
        self.add(vbox)

        self.create_ui()
        self.menubar = self.ui.get_widget('/Menubar')
        vbox.pack_start(self.menubar, expand=False)

        self.doc = document.Document()
        self.doc.set_brush(self.app.brush)
        self.tdw = tileddrawwidget.TiledDrawWidget(self.doc)
        vbox.pack_start(self.tdw)

        #self.zoomlevel_values = [0.09, 0.12,  0.18, 0.25, 0.33,  0.50, 0.66,  1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 8.0]
        self.zoomlevel_values = [            2.0/11, 0.25, 1.0/3, 0.50, 2.0/3, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 8.0]
        self.zoomlevel = self.zoomlevel_values.index(1.0)
        self.tdw.zoom_min = min(self.zoomlevel_values)
        self.tdw.zoom_max = max(self.zoomlevel_values)
        self.fullscreen = False

        self.popups = {
            'ChangeColorPopup': colorselectionwindow.ChangeColorPopup(self.app),
            'ColorWheelPopup': colorselectionwindow.ColorWheelPopup(self.app),
            }
        for w in self.popups.itervalues():
            w.connect("unmap-event", self.popup_unmap_cb)
        self.active_popup = None

        self.app.brush.settings_observers.append(self.brush_modified_cb)

        self.filename = None


    def create_ui(self):
        ag = gtk.ActionGroup('WindowActions')
        # FIXME: this xml menu only creates unneeded information duplication, I think.
		# FIXME: better just use glade...
        ui_string = """<ui>
          <menubar name='Menubar'>
            <menu action='FileMenu'>
              <menuitem action='New'/>
              <menuitem action='Open'/>
              <separator/>
              <menuitem action='Save'/>
              <menuitem action='SaveAs'/>
              <menuitem action='SaveNext'/>
              <separator/>
              <menuitem action='Quit'/>
            </menu>
            <menu action='EditMenu'>
              <menuitem action='Undo'/>
              <menuitem action='Redo'/>
              <separator/>
              <menuitem action='CopyLayer'/>
              <menuitem action='PasteLayer'/>
            </menu>
            <menu action='ViewMenu'>
              <menuitem action='Fullscreen'/>
              <separator/>
              <menuitem action='ZoomIn'/>
              <menuitem action='ZoomOut'/>
              <menuitem action='Zoom1'/>
              <separator/>
              <menuitem action='RotateRight'/>
              <menuitem action='RotateLeft'/>
              <menuitem action='Rotate0'/>
              <menuitem action='Flip'/>
              <separator/>
              <menuitem action='ViewHelp'/>
            </menu>
            <menu action='DialogMenu'>
              <menuitem action='BrushSelectionWindow'/>
              <menuitem action='BrushSettingsWindow'/>
              <menuitem action='ColorSelectionWindow'/>
              <menuitem action='SettingsWindow'/>
            </menu>
            <menu action='BrushMenu'>
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
                <separator/>
                <menuitem action='ContextHelp'/>
              </menu>
              <menuitem action='Bigger'/>
              <menuitem action='Smaller'/>
              <menuitem action='Brighter'/>
              <menuitem action='Darker'/>
              <menuitem action='MoreOpaque'/>
              <menuitem action='LessOpaque'/>
              <separator/>
              <menuitem action='Eraser'/>
              <separator/>
              <menuitem action='InvertColor'/>
              <menuitem action='PickColor'/>
              <menuitem action='ChangeColorPopup'/>
              <menuitem action='ColorWheelPopup'/>
              <menuitem action='ColorSelectionWindow'/>
            </menu>
            <menu action='LayerMenu'>
              <menuitem action='BackgroundWindow'/>
              <menuitem action='ClearLayer'/>
              <separator/>
              <menuitem action='NewLayerBG'/>
              <menuitem action='NewLayerFG'/>
              <menuitem action='RemoveLayer'/>
              <separator/>
              <menuitem action='LayerBG'/>
              <menuitem action='LayerFG'/>
              <menuitem action='PickLayer'/>
              <menuitem action='ToggleAbove'/>
            </menu>
            <menu action='DebugMenu'>
              <menuitem action='PrintInputs'/>
              <menuitem action='VisualizeRendering'/>
            </menu>
            <menu action='HelpMenu'>
              <menuitem action='Docu'/>
              <menuitem action='ShortcutHelp'/>
              <separator/>
              <menuitem action='About'/>
            </menu>
          </menubar>
        </ui>"""
        actions = [
			# name, stock id, label, accelerator, tooltip, callback
            ('FileMenu',     None, 'File'),
            ('New',          None, 'New', '<control>N', None, self.new_cb),
            ('Open',         None, 'Open...', '<control>O', None, self.open_cb),
            ('Save',         None, 'Save', '<control>S', None, self.save_cb),
            ('SaveAs',       None, 'Save As...', '<control><shift>S', None, self.save_as_cb),
            ('SaveNext',     None, 'Save as Scrap', 'F2', None, self.save_next_cb),
            ('Quit',         None, 'Quit', None, None, self.quit_cb),


            ('EditMenu',           None, 'Edit'),
            ('Undo',               None, 'Undo', '<control>Z', None, self.undo_cb),
            ('Redo',               None, 'Redo', '<control>Y', None, self.redo_cb),
            ('CopyLayer',          None, 'Copy Layer to Clipboard', '<control>C', None, self.copy_cb),
            ('PasteLayer',         None, 'Paste Layer from Clipboard', '<control>V', None, self.paste_cb),

            ('BrushMenu',    None, 'Brush'),
            ('InvertColor',  None, 'Invert Color', 'x', None, self.invert_color_cb),
            ('Brighter',     None, 'Brighter', None, None, self.brighter_cb),
            ('Darker',       None, 'Darker', None, None, self.darker_cb),
            ('Bigger',       None, 'Bigger', 'f', None, self.brush_bigger_cb),
            ('Smaller',      None, 'Smaller', 'd', None, self.brush_smaller_cb),
            ('MoreOpaque',   None, 'More Opaque', 's', None, self.more_opaque_cb),
            ('LessOpaque',   None, 'Less Opaque', 'a', None, self.less_opaque_cb),
            ('PickColor',    None, 'Pick Color', 'r', None, self.pick_color_cb),
            ('ChangeColorPopup', None, 'Change Color', 'v', None, self.popup_cb),
            ('ColorWheelPopup',  None, 'Color Wheel', None, None, self.popup_cb),
            ('Eraser',       None, 'Toggle Eraser Mode', 'e', None, self.eraser_cb),

            ('ContextMenu',  None, 'Brushkeys'),
            ('Context00',    None, 'Restore Brush 0', '0', None, self.context_cb),
            ('Context00s',   None, 'Save to Brush 0', '<control>0', None, self.context_cb),
            ('Context01',    None, 'Restore 1', '1', None, self.context_cb),
            ('Context01s',   None, 'Save 1', '<control>1', None, self.context_cb),
            ('Context02',    None, 'Restore 2', '2', None, self.context_cb),
            ('Context02s',   None, 'Save 2', '<control>2', None, self.context_cb),
            ('Context03',    None, 'Restore 3', '3', None, self.context_cb),
            ('Context03s',   None, 'Save 3', '<control>3', None, self.context_cb),
            ('Context04',    None, 'Restore 4', '4', None, self.context_cb),
            ('Context04s',   None, 'Save 4', '<control>4', None, self.context_cb),
            ('Context05',    None, 'Restore 5', '5', None, self.context_cb),
            ('Context05s',   None, 'Save 5', '<control>5', None, self.context_cb),
            ('Context06',    None, 'Restore 6', '6', None, self.context_cb),
            ('Context06s',   None, 'Save 6', '<control>6', None, self.context_cb),
            ('Context07',    None, 'Restore 7', '7', None, self.context_cb),
            ('Context07s',   None, 'Save 7', '<control>7', None, self.context_cb),
            ('Context08',    None, 'Restore 8', '8', None, self.context_cb),
            ('Context08s',   None, 'Save 8', '<control>8', None, self.context_cb),
            ('Context09',    None, 'Restore 9', '9', None, self.context_cb),
            ('Context09s',   None, 'Save 9', '<control>9', None, self.context_cb),
            ('ContextStore', None, 'Save to Most Recently Restored', 'q', None, self.context_cb),
            ('ContextHelp',  None, 'Help!', None, None, self.context_help_cb),

            ('LayerMenu',    None, 'Layers'),

            ('BackgroundWindow', None, 'Background Pattern...', None, None, self.toggleWindow_cb),
            ('ClearLayer',   None, 'Clear Layer', '<control>period', None, self.clear_layer_cb),
            ('PickLayer',    None, 'Select layer at cursor', 'h', None, self.pick_layer_cb),
            ('LayerBG',      None, 'Background (previous layer)', 'j', None, self.layer_bg_cb),
            ('LayerFG',      None, 'Foreground (next layer)',  'k', None, self.layer_fg_cb),
            ('NewLayerBG',   None, 'New Layer (behind current)', '<control>j', None, self.new_layer_cb),
            ('NewLayerFG',   None, 'New Layer (above current)', '<control>k', None, self.new_layer_cb),
            ('RemoveLayer',  None, 'Remove Layer', None, None, self.remove_layer_cb),
            ('ToggleAbove',  None, 'Toggle Layers Above Current', 'l', None, self.toggle_layers_above_cb),

            ('DialogMenu',  None, 'Windows'),
            ('BrushSelectionWindow',  None, 'Brush List', 'b', None, self.toggleWindow_cb),
            ('BrushSettingsWindow',   None, 'Brush Settings', '<control>b', None, self.toggleWindow_cb),
            ('ColorSelectionWindow',  None, 'GTK Color Dialog', 'g', None, self.toggleWindow_cb),
            ('SettingsWindow',        None, 'Settings', None, None, self.toggleWindow_cb),

            ('HelpMenu',     None, 'Help'),
            ('Docu', None, 'Where is the Documentation?', None, None, self.show_docu_cb),
            ('ShortcutHelp',  None, 'Change the Keyboard Shortcuts?', None, None, self.shortcut_help_cb),
            ('About', None, 'About MyPaint', None, None, self.show_about_cb),

            ('DebugMenu',    None, 'Debug'),


            ('ShortcutsMenu', None, 'Shortcuts'),

            ('ViewMenu', None, 'View'),
            ('Fullscreen',   None, 'Fullscreen', 'F11', None, self.fullscreen_cb),
            ('ZoomIn',       None, 'Zoom In', 'period', None, self.zoom_cb),
            ('ZoomOut',      None, 'Zoom Out', 'comma', None, self.zoom_cb),
            ('Zoom1',        None, 'Zoom 1:1', None, None, self.zoom_cb),
            ('RotateRight',  None, 'Rotate Clockwise', 'n', None, self.rotate_cb),
            ('RotateLeft',   None, 'Rotate Counterclockwise', 'm', None, self.rotate_cb),
            ('Rotate0',      None, 'Rotate Reset', None, None, self.rotate_cb),
            ('ViewHelp',     None, 'Help', None, None, self.view_help_cb),
            ]
        ag.add_actions(actions)
        toggle_actions = [
            # name, stock id, label, accelerator, tooltip, callback, default toggle status
            ('PrintInputs', None, 'Print Brush Input Values to stdout', None, None, self.print_inputs_cb),
            ('VisualizeRendering', None, 'Visualize Rendering', None, None, self.visualize_rendering_cb),
            ('Flip', None, 'Mirror Image', None, None, self.flip_cb),
            ]
        ag.add_toggle_actions(toggle_actions)
        self.ui = gtk.UIManager()
        self.ui.insert_action_group(ag, 0)
        self.ui.add_ui_from_string(ui_string)
        self.app.accel_group = self.ui.get_accel_group()
        self.add_accel_group(self.app.accel_group)

    def toggleWindow_cb(self, action):
        s = action.get_name()
        s = s[0].lower() + s[1:]
        w = getattr(self.app, s)
        if w.is_active():
            w.hide()
        else:
            w.show_all() # might be for the first time
            w.present()

    def print_inputs_cb(self, action):
        self.doc.brush.print_inputs = action.get_active()

    def visualize_rendering_cb(self, action):
        self.tdw.visualize_rendering = action.get_active()

    def start_profiling(self):
        def autopaint():
            import pylab
            events = pylab.load('painting30sec.dat.gz')
            events[:,0] *= 0.3
            events = list(events)
            t0 = time()
            t_old = 0.0
            for t, x, y, pressure in events:
                sleeptime = t-(time()-t0)
                if sleeptime > 0.001:
                    yield sleeptime
                dtime = t - t_old
                t_old = t
                self.doc.stroke_to(dtime, x, y, pressure)
            print 'replay done.'
            print self.repaints, 'repaints'
            gtk.main_quit()
            yield 10.0

        import gobject
        p = autopaint()
        def timer_cb():
            gobject.timeout_add(int(p.next()*1000.0), timer_cb)

        self.repaints = 0
        oldfunc=self.tdw.repaint
        def count_repaints(*args, **kwargs):
            self.repaints += 1
            return oldfunc(*args, **kwargs)
        self.tdw.repaint = count_repaints
        timer_cb()

        self.tdw.rotate(46.0/360*2*math.pi)
        
    def undo_cb(self, action):
        self.doc.undo()

        #cost = self.layer.rerender(only_estimate_cost=True)
        #if cost > 50:
        #    d = gtk.MessageDialog(
        #         type = gtk.MESSAGE_QUESTION,
        #         flags = gtk.DIALOG_MODAL,
        #         buttons = gtk.BUTTONS_YES_NO,
        #         message_format="This undo step will require %d brush strokes to be re-rendered. This might take some time.\n\nDo you really want to undo?" % cost
        #         )
        #    if d.run() != gtk.RESPONSE_YES:
        #        self.command_stack.redo()
        #    d.destroy()

        ## TODO: where does this code go?

    def redo_cb(self, action):
        self.doc.redo()

    def copy_cb(self, action):
        pixbuf = self.doc.layer.surface.render_as_pixbuf()
        cb = gtk.Clipboard()
        cb.set_image(pixbuf)

    def paste_cb(self, action):
        cb = gtk.Clipboard()
        def callback(clipboard, pixbuf, trash):
            if not pixbuf:
                print 'The clipboard doeas not contain any image to paste!'
                return
            self.doc.load_layer_from_pixbuf(pixbuf)
        cb.request_image(callback)

    def brush_modified_cb(self):
        # called at every brush setting modification, should return fast
        self.doc.set_brush(self.app.brush)

    def key_press_event_cb_before(self, win, event):
        key = event.keyval 
        #ANY_MODIFIER = gdk.SHIFT_MASK | gdk.MOD1_MASK | gdk.CONTROL_MASK
        #if event.state & ANY_MODIFIER:
        #    # allow user shortcuts with modifiers
        #    return False
        if key == keysyms.Left: 
            if event.state & gdk.CONTROL_MASK:
                self.rotate('RotateLeft')
            else:
                self.move('MoveLeft')
        elif key == keysyms.Right:
            if event.state & gdk.CONTROL_MASK:
                self.rotate('RotateRight')
            else:
                self.move('MoveRight')
        elif key == keysyms.Up  : self.move('MoveUp')
        elif key == keysyms.Down: self.move('MoveDown')
        elif key == keysyms.space: 
            if event.state & gdk.CONTROL_MASK:
                self.tdw.start_drag(self.dragfunc_rotate)
            else:
                Self.tdw.start_drag(self.dragfunc_translate)
        else: return False
        return True
    def key_release_event_cb_before(self, win, event):
        if event.keyval == keysyms.space:
            self.tdw.stop_drag(self.dragfunc_translate)
            self.tdw.stop_drag(self.dragfunc_rotate)
            return True
        return False

    def key_press_event_cb_after(self, win, event):
        # Not checking modifiers because this function gets only 
        # called if no user keybinding accepted the event.
        if event.keyval in [keysyms.KP_Add, keysyms.plus]: self.zoom('ZoomIn')
        elif event.keyval in [keysyms.KP_Subtract, keysyms.minus]: self.zoom('ZoomOut')
        elif self.fullscreen and event.keyval == keysyms.Escape: self.fullscreen_cb()
        else: return False
        return True
    def key_release_event_cb_after(self, win, event):
        return False

    def dragfunc_translate(self, dx, dy):
        self.tdw.scroll(-dx, -dy)

    def dragfunc_rotate(self, dx, dy):
        self.tdw.scroll(-dx, -dy, False)
        self.tdw.rotate(2*math.pi*dx/500.0)

    #def dragfunc_rotozoom(self, dx, dy):
    #    self.tdw.scroll(-dx, -dy, False)
    #    self.tdw.zoom(math.exp(-dy/100.0))
    #    self.tdw.rotate(2*math.pi*dx/500.0)

    def button_press_cb(self, win, event):
        #print event.device, event.button
        if event.button == 2:
            self.tdw.start_drag(self.dragfunc_translate)
        elif event.button == 3:
            self.tdw.start_drag(self.dragfunc_rotate)

    def button_release_cb(self, win, event):
        #print event.device, event.button
        if event.button == 2:
            self.tdw.stop_drag(self.dragfunc_translate)
        elif event.button == 3:
            self.tdw.stop_drag(self.dragfunc_rotate)

    def scroll_cb(self, win, event):
        d = event.direction
        if event.state & gdk.CONTROL_MASK:
            if d == gdk.SCROLL_UP:
                self.zoom('ZoomIn')
            elif d == gdk.SCROLL_DOWN:
                self.zoom('ZoomOut')
        else:
            mapping = {
                gdk.SCROLL_RIGHT: 'MoveRight',
                gdk.SCROLL_LEFT: 'MoveLeft',
                gdk.SCROLL_UP: 'MoveUp',
                gdk.SCROLL_DOWN: 'MoveDown'
            }

            if event.state & gdk.SHIFT_MASK:
                # remap up and down to left and right so that it's
                # possible to scroll easier with mouse + keyboard.
                mapping.update({
                    gdk.SCROLL_UP: 'MoveLeft',
                    gdk.SCROLL_DOWN: 'MoveRight'
                })
            self.move(mapping[d])

    def clear_layer_cb(self, action):
        self.doc.clear_layer()
        
    def remove_layer_cb(self, action):
        if len(self.doc.layers) == 1:
            self.doc.clear_layer()
        else:
            self.doc.remove_layer()

    def layer_bg_cb(self, action):
        idx = self.doc.layer_idx - 1
        if idx < 0: return
        self.doc.select_layer(idx)

    def layer_fg_cb(self, action):
        idx = self.doc.layer_idx + 1
        if idx >= len(self.doc.layers): return
        self.doc.select_layer(idx)

    def pick_layer_cb(self, action):
        x, y = self.tdw.get_cursor_in_model_coordinates()
        for idx, layer in reversed(list(enumerate(self.doc.layers))):
            alpha = layer.surface.get_alpha (x, y, 5)
            if alpha > 0.1:
                self.doc.select_layer(idx)
                return
        self.doc.select_layer(0)

    def new_layer_cb(self, action):
        insert_idx = self.doc.layer_idx
        if action.get_name() == 'NewLayerFG':
            insert_idx += 1
        self.doc.add_layer(insert_idx)

    def toggle_layers_above_cb(self, action):
        self.tdw.toggle_show_layers_above()

    def invert_color_cb(self, action):
        self.end_eraser_mode()
        self.app.brush.invert_color()
        
    def pick_color_cb(self, action):
        self.end_eraser_mode()
        size = int(self.app.brush.get_actual_radius() * math.sqrt(math.pi))
        if size < 1:
            size = 1
        self.app.colorSelectionWindow.pick_color_at_pointer(size)

    def popup_cb(self, action):
        self.end_eraser_mode()
        self.popup(action.get_name())
    def popup(self, name):
        w = self.popups[name]
        if w is self.active_popup:
            # pressed the key for the same popup which is already active
            w.hide()
            transitions = {
                'ChangeColorPopup': 'ColorWheelPopup',
                'ColorWheelPopup': 'ChangeColorPopup',
                }
            if name in transitions:
                self.popup(transitions[name])
        else:
            if self.active_popup:
                self.active_popup.hide()
            self.active_popup = w
            w.popup()

    def popup_unmap_cb(self, widget, event):
        if self.active_popup is widget:
            self.active_popup = None

    def eraser_cb(self, action):
        adj = self.app.brush_adjustment['eraser']
        if adj.get_value() > 0.9:
            adj.set_value(0.0)
        else:
            adj.set_value(1.0)

    def end_eraser_mode(self):
        adj = self.app.brush_adjustment['eraser']
        if adj.get_value() > 0.9:
            adj.set_value(0.0)

    def brush_bigger_cb(self, action):
        adj = self.app.brush_adjustment['radius_logarithmic']
        adj.set_value(adj.get_value() + 0.3)
    def brush_smaller_cb(self, action):
        adj = self.app.brush_adjustment['radius_logarithmic']
        adj.set_value(adj.get_value() - 0.3)

    def more_opaque_cb(self, action):
        # FIXME: hm, looks this slider should be logarithmic?
        adj = self.app.brush_adjustment['opaque']
        adj.set_value(adj.get_value() * 1.8)
    def less_opaque_cb(self, action):
        adj = self.app.brush_adjustment['opaque']
        adj.set_value(adj.get_value() / 1.8)

    def brighter_cb(self, action):
        self.end_eraser_mode()
        h, s, v = self.app.brush.get_color_hsv()
        v += 0.08
        if v > 1.0: v = 1.0
        self.app.brush.set_color_hsv((h, s, v))
    def darker_cb(self, action):
        self.end_eraser_mode()
        h, s, v = self.app.brush.get_color_hsv()
        v -= 0.08
        if v < 0.0: v = 0.0
        self.app.brush.set_color_hsv((h, s, v))
        
    def open_file(self, filename):
        try:
            # TODO: that would be "open_file_as_layer"
            #pixbuf = gdk.pixbuf_new_from_file(filename)
            #cmd = command.LoadImage(self.layer, pixbuf)
            #self.doc.execute(cmd)
            self.doc.load(filename)
        except Exception, e:
            d = gtk.MessageDialog(self, type=gtk.MESSAGE_ERROR, buttons=gtk.BUTTONS_OK)
            d.set_markup(str(e))
            d.run()
            d.destroy()
            raise
        else:
            print 'Loaded from', filename
            self.filename = filename
            self.zoom('Zoom1')
            self.rotate('Rotate0')
            self.tdw.recenter_document()

    def save_file(self, filename):
        self.filename = filename
        try:
            self.doc.save(filename)
        except Exception, e:
            print e
            d = gtk.MessageDialog(self, type=gtk.MESSAGE_ERROR, buttons=gtk.BUTTONS_OK)
            d.set_markup(str(e))
            d.run()
            d.destroy()
            print 'Failed to save!'
            raise
        else:
            print 'Saved to ' + filename


    def confirm_destructive_action(self, title='Confirm', question='Really continue?'):
        #t = self.get_unsaved_painting_time()
        t = self.doc.get_total_painting_time()
        if t < 60:
            # no need to ask
            return True

        if t > 120:
            t = '%d minutes' % (t/60)
        else:
            t = '%d seconds' % t
        d = gtk.MessageDialog(type = gtk.MESSAGE_QUESTION,
                              buttons = gtk.BUTTONS_YES_NO,
                              flags = gtk.DIALOG_MODAL,
                              )
        d.set_title(title)
        d.set_markup("<b>" + question + "</b>\n\nThis will discard %s of unsaved painting." % t)
        response = d.run()
        d.destroy()
        return response == gtk.RESPONSE_YES

    def new_cb(self, action):
        if not self.confirm_destructive_action():
            return
        self.doc.clear()
        self.filename = None

    def add_file_filters(self, dialog):
        f = gtk.FileFilter()
        f.set_name("Any Format (*.png; *.ora)")
        f.add_pattern("*.png")
        f.add_pattern("*.ora")
        dialog.add_filter(f)

        f = gtk.FileFilter()
        f.set_name("PNG without layers (*.png)")
        f.add_pattern("*.png")
        dialog.add_filter(f)

        f = gtk.FileFilter()
        f.set_name("OpenRaster (*.ora)")
        f.add_pattern("*.ora")
        dialog.add_filter(f)

        #f = gtk.FileFilter()
        #f.set_name("MyPaint (*.myp)")
        #f.add_pattern("*.myp")
        #dialog.add_filter(f)

    def open_cb(self, action):
        if not self.confirm_destructive_action():
            return
        dialog = gtk.FileChooserDialog("Open..", self,
                                       gtk.FILE_CHOOSER_ACTION_OPEN,
                                       (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                        gtk.STOCK_OPEN, gtk.RESPONSE_OK))
        dialog.set_default_response(gtk.RESPONSE_OK)
        self.add_file_filters(dialog)

        if self.filename:
            dialog.set_filename(self.filename)
        try:
            if dialog.run() == gtk.RESPONSE_OK:
                self.open_file(dialog.get_filename())
        finally:
            dialog.destroy()
        
    def save_cb(self, action):
        if not self.filename:
            self.save_as_cb(action)
        else:
            self.save_file(self.filename)

    def save_as_cb(self, action):
        dialog = gtk.FileChooserDialog("Save..", self,
                                       gtk.FILE_CHOOSER_ACTION_SAVE,
                                       (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                        gtk.STOCK_SAVE, gtk.RESPONSE_OK))
        dialog.set_default_response(gtk.RESPONSE_OK)
        self.add_file_filters(dialog)

        if self.filename:
            dialog.set_filename(self.filename)
        try:
            if dialog.run() == gtk.RESPONSE_OK:
                filename = dialog.get_filename()
                trash, ext = os.path.splitext(filename)
                if not ext:
                    filename += '.png'
                    #filename += '.myp'
                    # TODO: auto-propose .ora when using layers or non-solid background patterns
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
                    self.save_file(filename)
        finally:
            dialog.destroy()

    def save_next_cb(self, action):
        filename = self.filename
        if filename:
            while True:
                # append a letter
                name, ext = os.path.splitext(filename)
                letter = 'a'
                if len(name) > 2 and name[-2] == '_' and name[-1] >= 'a' and name[-1] < 'z':
                    letter = chr(ord(name[-1]) + 1)
                    name = name[:-2]
                name = name + '_' + letter
                filename = name + '.png'
                if not os.path.exists(filename):
                    break
        else:
            # we don't have a filename yet
            prefix = self.app.settingsWindow.save_next_prefix
            maximum = 0
            for filename in glob(prefix + '[0-9][0-9][0-9]*'):
                filename = filename[len(prefix):]
                res = re.findall(r'[0-9]*', filename)
                if not res: continue
                number = int(res[0])
                if number > maximum:
                    maximum = number
            filename = '%s%03d.png' % (prefix, maximum+1)

        assert not os.path.exists(filename)
        self.save_file(filename)

    def quit_cb(self, *trash):
        #self.finish_pending_actions()
        self.doc.split_stroke()
        self.app.save_gui_config() # FIXME: should do this periodically, not only on quit

        if not self.confirm_destructive_action(title='Quit', question='Really Quit?'):
            return True

        gtk.main_quit()
        return False

    def zoom_cb(self, action):
        self.zoom(action.get_name())
    def rotate_cb(self, action):
        self.rotate(action.get_name())
    def flip_cb(self, action):
        self.tdw.set_flipped(action.get_active())

    def move(self, command):
        self.doc.split_stroke()
        step = min(self.tdw.window.get_size()) / 5
        if   command == 'MoveLeft' : self.tdw.scroll(-step, 0)
        elif command == 'MoveRight': self.tdw.scroll(+step, 0)
        elif command == 'MoveUp'   : self.tdw.scroll(0, -step)
        elif command == 'MoveDown' : self.tdw.scroll(0, +step)
        else: assert 0

    def zoom(self, command):
        if   command == 'ZoomIn' : self.zoomlevel += 1
        elif command == 'ZoomOut': self.zoomlevel -= 1
        elif command == 'Zoom1'  : self.zoomlevel = self.zoomlevel_values.index(1.0)
        else: assert 0
        if self.zoomlevel < 0: self.zoomlevel = 0
        if self.zoomlevel >= len(self.zoomlevel_values): self.zoomlevel = len(self.zoomlevel_values) - 1
        z = self.zoomlevel_values[self.zoomlevel]
        self.tdw.set_zoom(z)

    def rotate(self, command):
        if   command == 'RotateRight': self.tdw.rotate(+2*math.pi/14)
        elif command == 'RotateLeft' : self.tdw.rotate(-2*math.pi/14)
        elif command == 'Rotate0'    : self.tdw.set_rotation(0.0)
        else: assert 0

    def fullscreen_cb(self, *trash):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.menubar.hide()
            self.window.fullscreen()
            self.tdw.set_scroll_at_edges(True)
        else:
            self.window.unfullscreen()
            self.menubar.show()
            self.tdw.set_scroll_at_edges(False)

    def context_cb(self, action):
        # TODO: this context-thing is not very useful like that, is it?
        #       You overwrite your settings too easy by accident.
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
            context.save()
        else: # restore
            self.app.select_brush(context)
            self.app.brushSelectionWindow.set_preview_pixbuf(context.preview)

    def show_about_cb(self, action):
        d = gtk.MessageDialog(self, buttons=gtk.BUTTONS_OK)

        d.set_markup(
            u"MyPaint %s - pressure sensitive painting application\n"
            u"Copyright (C) 2005-2008\n"
            u"Martin Renold &lt;martinxyz@gmx.ch&gt;\n\n"
            u"Contributors:\n"
            u"Artis Rozentāls &lt;artis@aaa.apollo.lv&gt; (brushes)\n"
            u"Yves Combe &lt;yves@ycombe.net&gt; (portability)\n"
            u"Sebastian Kraft (desktop icon)\n"
            u"Popolon &lt;popolon@popolon.org&gt; (brushes)\n"
            u"Clement Skau &lt;clementskau@gmail.com&gt; (programming)\n"
            u"\n"
            u"This program is free software; you can redistribute it and/or modify "
            u"it under the terms of the GNU General Public License as published by "
            u"the Free Software Foundation; either version 2 of the License, or "
            u"(at your option) any later version.\n"
            u"\n"
            u"This program is distributed in the hope that it will be useful, "
            u"but WITHOUT ANY WARRANTY. See the COPYING file for more details."
            % MYPAINT_VERSION
            )

        d.run()
        d.destroy()

    def show_docu_cb(self, action):
        d = gtk.MessageDialog(self, buttons=gtk.BUTTONS_OK)
        d.set_markup("There is a tutorial in the html directory, also available "
                     "on the MyPaint homepage. It explains the features which are "
                     "hard to discover yourself.\n\n"
                     "Comments about the brush settings (opaque, hardness, etc.) and "
                     "inputs (pressure, speed, etc.) are available as tooltips. "
                     "Put your mouse over a label to see them. "
                     "\n"
                     )
        d.run()
        d.destroy()

    def context_help_cb(self, action):
        d = gtk.MessageDialog(self, buttons=gtk.BUTTONS_OK)
        d.set_markup("This is used to quickly save/restore brush settings "
                     "using keyboard shortcuts. You can paint with one hand and "
                     "change brushes with the other without interrupting."
                     "\n\n"
                     "There are 10 memory slots to hold brush settings.\n"
                     "Those are annonymous "
                     "brushes, they are not visible in the brush selector list. "
                     "But they will stay even if you quit. "
                     "They will also remember the selected color. In contrast, selecting a "
                     "normal brush never changes the color. "
                     )
        d.run()
        d.destroy()

    def shortcut_help_cb(self, action):
        d = gtk.MessageDialog(self, buttons=gtk.BUTTONS_OK)
        d.set_markup("Move your mouse over a menu entry, then press the key to "
                     "assign.")
        d.run()
        d.destroy()

    def view_help_cb(self, action):
        d = gtk.MessageDialog(self, buttons=gtk.BUTTONS_OK)
        d.set_markup(
            "You can also drag the canvas with the mouse while holding the middle mouse button or spacebar. "
            "or with the arrow keys. You can rotate by holding the right mouse button."
            "\n\n"
            "In contrast to earlier versions, scrolling and zooming are harmless now and "
            "will not make you run out of memory. But you still require a lot of memory "
            "if you paint all over while fully zoomed out."
            )
        d.run()
        d.destroy()

