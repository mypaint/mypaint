# -*- coding: utf-8 -*-
#
# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""
This is the main drawing window, containing menu actions.
Painting is done in tileddrawwidget.py.
"""

MYPAINT_VERSION="0.7.1"

import os, re, math
from time import time
from glob import glob
import traceback

import gtk
from gtk import gdk, keysyms

import tileddrawwidget, colorselectionwindow, historypopup, \
       stategroup, keyboard
from lib import document, helpers

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

        self.doc = document.Document()
        self.doc.set_brush(self.app.brush)

        self.create_ui()
        self.menubar = self.ui.get_widget('/Menubar')
        vbox.pack_start(self.menubar, expand=False)

        self.tdw = tileddrawwidget.TiledDrawWidget(self.doc)
        vbox.pack_start(self.tdw)

        # FIXME: hack, to be removed
        filename = os.path.join(self.app.datapath, 'backgrounds', '03_check1.png')
        pixbuf = gdk.pixbuf_new_from_file(filename)
        self.tdw.neutral_background_pixbuf = helpers.gdkpixbuf2numpy(pixbuf)

        self.zoomlevel_values = [2.0/11, 0.25, 1.0/3, 0.50, 2.0/3, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 8.0]
        self.zoomlevel = self.zoomlevel_values.index(1.0)
        self.tdw.zoom_min = min(self.zoomlevel_values)
        self.tdw.zoom_max = max(self.zoomlevel_values)
        self.fullscreen = False

        self.app.brush.settings_observers.append(self.brush_modified_cb)
        self.tdw.device_observers.append(self.device_changed_cb)
            
        historyfile_name = os.path.join(self.app.confpath, 'save_history.conf')
        if os.path.exists(historyfile_name):
            self.save_history = [line.strip() for line in open(historyfile_name)]
        else:
            self.save_history = []

        self.init_save_dialog()

         #filename is a property so that all changes will update the title
        self._filename = None
        
        
    def get_filename(self):
        return self._filename 
    def set_filename(self,value):
        self._filename = value
        if self.filename: 
            self.set_title("Mypaint - %s" % self.filename)
        else:
            self.set_title("Mypaint")
    filename = property(get_filename, set_filename)

    def create_ui(self):
        ag = self.action_group = gtk.ActionGroup('WindowActions')
        # FIXME: this xml menu only creates unneeded information duplication, I think.
		# FIXME: better just use glade...
        ui_string = """<ui>
          <menubar name='Menubar'>
            <menu action='FileMenu'>
              <menuitem action='New'/>
              <menuitem action='Open'/>
              <menuitem action='OpenRecent'/>
              <separator/>
              <menuitem action='Save'/>
              <menuitem action='SaveAs'/>
              <menuitem action='SaveScrap'/>
              <separator/>
              <menuitem action='Quit'/>
            </menu>
            <menu action='EditMenu'>
              <menuitem action='Undo'/>
              <menuitem action='Redo'/>
              <separator/>
              <menuitem action='CopyLayer'/>
              <menuitem action='PasteLayer'/>
              <separator/>
              <menuitem action='SettingsWindow'/>
            </menu>
            <menu action='ViewMenu'>
              <menuitem action='Fullscreen'/>
              <separator/>
              <menuitem action='ResetView'/>
              <menuitem action='ZoomIn'/>
              <menuitem action='ZoomOut'/>
              <menuitem action='RotateLeft'/>
              <menuitem action='RotateRight'/>
              <menuitem action='Flip'/>
              <separator/>
              <menuitem action='ViewHelp'/>
            </menu>
            <menu action='BrushMenu'>
              <menuitem action='BrushSelectionWindow'/>
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
              <separator/>
              <menuitem action='BrushSettingsWindow'/>
              <separator/>
              <menuitem action='Bigger'/>
              <menuitem action='Smaller'/>
              <menuitem action='MoreOpaque'/>
              <menuitem action='LessOpaque'/>
              <separator/>
              <menuitem action='Eraser'/>
            </menu>
            <menu action='ColorMenu'>
              <menuitem action='ColorSelectionWindow'/>
              <menuitem action='ColorWheelPopup'/>
              <menuitem action='ChangeColorPopup'/>
              <menuitem action='PickColor'/>
              <menuitem action='ColorHistoryPopup'/>
              <separator/>
              <menuitem action='Brighter'/>
              <menuitem action='Darker'/>
            </menu>
            <menu action='LayerMenu'>
              <menuitem action='BackgroundWindow'/>
              <menuitem action='ClearLayer'/>
              <separator/>
              <menuitem action='NewLayerFG'/>
              <menuitem action='NewLayerBG'/>
              <menuitem action='RemoveLayer'/>
              <menuitem action='MergeLayer'/>
              <separator/>
              <menuitem action='PickLayer'/>
              <menuitem action='LayerFG'/>
              <menuitem action='LayerBG'/>
              <menuitem action='SoloLayer'/>
              <menuitem action='ToggleAbove'/>
            </menu>
            <menu action='DebugMenu'>
              <menuitem action='PrintInputs'/>
              <menuitem action='VisualizeRendering'/>
              <menuitem action='NoDoubleBuffereing'/>
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
            ('OpenRecent',   None, 'Open Recent', 'F3', None, self.open_recent_cb),
            ('Save',         None, 'Save', '<control>S', None, self.save_cb),
            ('SaveAs',       None, 'Save As...', '<control><shift>S', None, self.save_as_cb),
            ('SaveScrap',    None, 'Save Next Scrap', 'F2', None, self.save_scrap_cb),
            ('Quit',         None, 'Quit', '<control>q', None, self.quit_cb),


            ('EditMenu',           None, 'Edit'),
            ('Undo',               None, 'Undo', 'Z', None, self.undo_cb),
            ('Redo',               None, 'Redo', 'Y', None, self.redo_cb),
            ('CopyLayer',          None, 'Copy Layer to Clipboard', '<control>C', None, self.copy_cb),
            ('PasteLayer',         None, 'Paste Layer from Clipboard', '<control>V', None, self.paste_cb),

            ('BrushMenu',    None, 'Brush'),
            ('Brighter',     None, 'Brighter', None, None, self.brighter_cb),
            ('Smaller',      None, 'Smaller', 'd', None, self.brush_smaller_cb),
            ('MoreOpaque',   None, 'More Opaque', 's', None, self.more_opaque_cb),
            ('LessOpaque',   None, 'Less Opaque', 'a', None, self.less_opaque_cb),
            ('Eraser',       None, 'Toggle Eraser Mode', 'e', None, self.eraser_cb),

            ('ColorMenu',    None, 'Color'),
            ('Darker',       None, 'Darker', None, None, self.darker_cb),
            ('Bigger',       None, 'Bigger', 'f', None, self.brush_bigger_cb),
            ('PickColor',    None, 'Pick Color', 'r', None, self.pick_color_cb),
            ('ColorHistoryPopup',  None, 'Color History', 'x', None, self.popup_cb),
            ('ChangeColorPopup', None, 'Change Color', 'v', None, self.popup_cb),
            ('ColorWheelPopup',  None, 'Color Wheel', None, None, self.popup_cb),

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

            ('BackgroundWindow', None, 'Background...', None, None, self.toggleWindow_cb),
            ('ClearLayer',   None, 'Clear', 'Delete', None, self.clear_layer_cb),
            ('PickLayer',    None, 'Select Layer at Cursor', 'h', None, self.pick_layer_cb),
            ('LayerFG',      None, 'Next (above current)',  'Page_Up', None, self.layer_fg_cb),
            ('LayerBG',      None, 'Next (below current)', 'Page_Down', None, self.layer_bg_cb),
            ('NewLayerFG',   None, 'New (above current)', '<control>Page_Up', None, self.new_layer_cb),
            ('NewLayerBG',   None, 'New (below current)', '<control>Page_Down', None, self.new_layer_cb),
            ('MergeLayer',   None, 'Merge Down', '<control>Delete', None, self.merge_layer_cb),
            ('RemoveLayer',  None, 'Remove', '<shift>Delete', None, self.remove_layer_cb),
            ('SoloLayer',    None, 'Toggle Solo Mode', 'Home', None, self.solo_layer_cb),
            ('ToggleAbove',  None, 'Toggle Layers Above Current', 'End', None, self.toggle_layers_above_cb), # TODO: make toggle action

            ('BrushSelectionWindow',  None, 'Brush List...', 'b', None, self.toggleWindow_cb),
            ('BrushSettingsWindow',   None, 'Brush Settings...', '<control>b', None, self.toggleWindow_cb),
            ('ColorSelectionWindow',  None, 'Color Triangle...', 'g', None, self.toggleWindow_cb),
            ('SettingsWindow',        None, 'Settings...', None, None, self.toggleWindow_cb),

            ('HelpMenu',     None, 'Help'),
            ('Docu', None, 'Where is the Documentation?', None, None, self.show_docu_cb),
            ('ShortcutHelp',  None, 'Change the Keyboard Shortcuts?', None, None, self.shortcut_help_cb),
            ('About', None, 'About MyPaint', None, None, self.show_about_cb),

            ('DebugMenu',    None, 'Debug'),


            ('ShortcutsMenu', None, 'Shortcuts'),

            ('ViewMenu', None, 'View'),
            ('Fullscreen',   None, 'Fullscreen', 'F11', None, self.fullscreen_cb),
            ('ResetView',   None, 'Reset (Zoom, Rotation, Mirror)', None, None, self.reset_view_cb),
            ('ZoomOut',      None, 'Zoom Out (at cursor)', 'comma', None, self.zoom_cb),
            ('ZoomIn',       None, 'Zoom In', 'period', None, self.zoom_cb),
            ('RotateLeft',   None, 'Rotate Counterclockwise', None, None, self.rotate_cb),
            ('RotateRight',  None, 'Rotate Clockwise', None, None, self.rotate_cb),
            ('ViewHelp',     None, 'Help', None, None, self.view_help_cb),
            ]
        ag.add_actions(actions)
        toggle_actions = [
            # name, stock id, label, accelerator, tooltip, callback, default toggle status
            ('PrintInputs', None, 'Print Brush Input Values to stdout', None, None, self.print_inputs_cb),
            ('VisualizeRendering', None, 'Visualize Rendering', None, None, self.visualize_rendering_cb),
            ('NoDoubleBuffereing', None, 'Disable GTK Double Buffering', None, None, self.no_double_buffering_cb),
            ('Flip', None, 'Mirror Image', 'i', None, self.flip_cb),
            ]
        ag.add_toggle_actions(toggle_actions)
        self.ui = gtk.UIManager()
        self.ui.insert_action_group(ag, 0)
        self.ui.add_ui_from_string(ui_string)
        #self.app.accel_group = self.ui.get_accel_group()

        self.app.kbm = kbm = keyboard.KeyboardManager()
        # TODO: and now tell the keyboard manager about hardcoded keys ("unless used otherwise" aliases)
        kbm.add_window(self)

        for action in ag.list_actions():
            self.app.kbm.takeover_action(action)

        kbm.add_extra_key('<control>z', 'Undo')
        kbm.add_extra_key('<control>y', 'Redo')
        kbm.add_extra_key('KP_Add', 'ZoomIn')
        kbm.add_extra_key('KP_Subtract', 'ZoomOut')

        kbm.add_extra_key('Left', lambda(action): self.move('MoveLeft'))
        kbm.add_extra_key('Right', lambda(action): self.move('MoveRight'))
        kbm.add_extra_key('Down', lambda(action): self.move('MoveDown'))
        kbm.add_extra_key('Up', lambda(action): self.move('MoveUp'))

        kbm.add_extra_key('<control>Left', 'RotateLeft')
        kbm.add_extra_key('<control>Right', 'RotateRight')

        sg = stategroup.StateGroup()
        self.layerblink_state = sg.create_state(self.layerblink_state_enter, self.layerblink_state_leave)

        # separate stategroup...
        sg2 = stategroup.StateGroup()
        self.layersolo_state = sg2.create_state(self.layersolo_state_enter, self.layersolo_state_leave)
        self.layersolo_state.autoleave_timeout = None

        p2s = sg.create_popup_state
        changer = p2s(colorselectionwindow.ChangeColorPopup(self.app))
        wheel = p2s(colorselectionwindow.ColorWheelPopup(self.app))
        hist = p2s(historypopup.HistoryPopup(self.app, self.doc))

        self.popup_states = {
            'ChangeColorPopup': changer,
            'ColorWheelPopup': wheel,
            'ColorHistoryPopup': hist,
            }
        changer.next_state = wheel
        wheel.next_state = changer
        changer.autoleave_timeout = None
        wheel.autoleave_timeout = None
        hist.autoleave_timeout = None
        hist.autoleave_timeout = 0.600
        self.history_popup_state = hist

    def toggleWindow_cb(self, action):
        s = action.get_name()
        s = s[0].lower() + s[1:]
        w = getattr(self.app, s)
        if w.window and w.window.is_visible():
            w.hide()
        else:
            w.show_all() # might be for the first time
            w.present()

    def print_inputs_cb(self, action):
        self.doc.brush.print_inputs = action.get_active()

    def visualize_rendering_cb(self, action):
        self.tdw.visualize_rendering = action.get_active()
    def no_double_buffering_cb(self, action):
        self.tdw.set_double_buffered(not action.get_active())

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

    def redo_cb(self, action):
        self.doc.redo()

    def copy_cb(self, action):
        # use the full document bbox, so we can past layers back to the correct position
        bbox = self.doc.get_bbox()
        pixbuf = self.doc.layer.surface.render_as_pixbuf(*bbox)
        cb = gtk.Clipboard()
        cb.set_image(pixbuf)

    def paste_cb(self, action):
        cb = gtk.Clipboard()
        def callback(clipboard, pixbuf, trash):
            if not pixbuf:
                print 'The clipboard doeas not contain any image to paste!'
                return
            # paste to the upper left of our doc bbox (see above)
            x, y, w, h = self.doc.get_bbox()
            self.doc.load_layer_from_pixbuf(pixbuf, x, y)
        cb.request_image(callback)

    def brush_modified_cb(self):
        # called at every brush setting modification, should return fast
        self.doc.set_brush(self.app.brush)

    def key_press_event_cb_before(self, win, event):
        key = event.keyval 
        ctrl = event.state & gdk.CONTROL_MASK
        #ANY_MODIFIER = gdk.SHIFT_MASK | gdk.MOD1_MASK | gdk.CONTROL_MASK
        #if event.state & ANY_MODIFIER:
        #    # allow user shortcuts with modifiers
        #    return False
        if key == keysyms.space:
            if ctrl:
                self.tdw.start_drag(self.dragfunc_rotate)
            else:
                self.tdw.start_drag(self.dragfunc_translate)
        else: return False
        return True
    def key_release_event_cb_before(self, win, event):
        if event.keyval == keysyms.space:
            self.tdw.stop_drag(self.dragfunc_translate)
            self.tdw.stop_drag(self.dragfunc_rotate)
            return True
        return False

    def key_press_event_cb_after(self, win, event):
        key = event.keyval
        ctrl = event.state & gdk.CONTROL_MASK
        if self.fullscreen and key == keysyms.Escape: self.fullscreen_cb()
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
        if event.type != gdk.BUTTON_PRESS:
            # ignore the extra double-click event
            return
        if event.button == 2:
            # check whether we are painting (accidental)
            pressure = event.get_axis(gdk.AXIS_PRESSURE)
            if (event.state & gdk.BUTTON1_MASK) or pressure:
                # do not allow dragging while painting (often happens accidentally)
                pass
            else:
                self.tdw.start_drag(self.dragfunc_translate)
        elif event.button == 1:
            # pick color "standard"; TODO: show color picking cursor?
            if event.state & gdk.CONTROL_MASK:
                self.pick_color_cb(None)
        elif event.button == 3:
            self.history_popup_state.activate()

    def button_release_cb(self, win, event):
        #print event.device, event.button
        if event.button == 2:
            self.tdw.stop_drag(self.dragfunc_translate)
        # too slow to be useful:
        #elif event.button == 3:
        #    self.tdw.stop_drag(self.dragfunc_rotate)

    def scroll_cb(self, win, event):
        d = event.direction
        if d == gdk.SCROLL_UP:
            if event.state & gdk.SHIFT_MASK:
                self.rotate('RotateLeft')
            else:
                self.zoom('ZoomIn')
        elif d == gdk.SCROLL_DOWN:
            if event.state & gdk.SHIFT_MASK:
                self.rotate('RotateRight')
            else:
                self.zoom('ZoomOut')
        elif d == gdk.SCROLL_LEFT:
            self.rotate('RotateRight')
        elif d == gdk.SCROLL_LEFT:
            self.rotate('RotateLeft')

    def clear_layer_cb(self, action):
        self.doc.clear_layer()
        if len(self.doc.layers) == 1:
            # this is like creating a new document:
            # make "save next" use a new file name
            self.filename = None
        
    def remove_layer_cb(self, action):
        if len(self.doc.layers) == 1:
            self.doc.clear_layer()
        else:
            self.doc.remove_layer()
            self.layerblink_state.activate(action)

    def layer_bg_cb(self, action):
        idx = self.doc.layer_idx - 1
        if idx < 0:
            return
        self.doc.select_layer(idx)
        self.layerblink_state.activate(action)

    def layer_fg_cb(self, action):
        idx = self.doc.layer_idx + 1
        if idx >= len(self.doc.layers):
            return
        self.doc.select_layer(idx)
        self.layerblink_state.activate(action)

    def pick_layer_cb(self, action):
        x, y = self.tdw.get_cursor_in_model_coordinates()
        for idx, layer in reversed(list(enumerate(self.doc.layers))):
            alpha = layer.surface.get_alpha (x, y, 5)
            if alpha > 0.1:
                self.doc.select_layer(idx)
                self.layerblink_state.activate(action)
                return
        self.doc.select_layer(0)
        self.layerblink_state.activate(action)

    def layerblink_state_enter(self):
        self.tdw.current_layer_solo = True
        self.tdw.queue_draw()
    def layerblink_state_leave(self, reason):
        if self.layersolo_state.active:
            # FIXME: use state machine concept, maybe?
            return
        self.tdw.current_layer_solo = False
        self.tdw.queue_draw()
    def layersolo_state_enter(self):
        s = self.layerblink_state
        if s.active:
            s.leave()
        self.tdw.current_layer_solo = True
        self.tdw.queue_draw()
    def layersolo_state_leave(self, reason):
        self.tdw.current_layer_solo = False
        self.tdw.queue_draw()

    #def blink_layer_cb(self, action):
    #    self.layerblink_state.activate(action)

    def solo_layer_cb(self, action):
        self.layersolo_state.toggle(action)

    def new_layer_cb(self, action):
        insert_idx = self.doc.layer_idx
        if action.get_name() == 'NewLayerFG':
            insert_idx += 1
        self.doc.add_layer(insert_idx)
        self.layerblink_state.activate(action)

    def merge_layer_cb(self, action):
        dst_idx = self.doc.layer_idx - 1
        if dst_idx < 0:
            return
        self.doc.merge_layer(dst_idx)
        self.layerblink_state.activate(action)

    def toggle_layers_above_cb(self, action):
        self.tdw.toggle_show_layers_above()

    def pick_color_cb(self, action):
        self.end_eraser_mode()
        size = int(self.app.brush.get_actual_radius() * math.sqrt(math.pi))
        if size < 1:
            size = 1
        self.app.colorSelectionWindow.pick_color_at_pointer(size)


    def popup_cb(self, action):
        # This doesn't really belong here...
        # ...maybe should eraser_mode a GUI state too?
        self.end_eraser_mode()

        state = self.popup_states[action.get_name()]
        state.activate(action)

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

    def device_changed_cb(self, old_device, new_device):
        # just enable eraser mode for now (TODO: remember full tool settings)
        # small problem with this code: it doesn't work well with brushes that have (eraser not in [1.0, 0.0])
        adj = self.app.brush_adjustment['eraser']
        if old_device is None and new_device.source != gdk.SOURCE_ERASER:
            # keep whatever startup brush was choosen
            return
        if new_device.source == gdk.SOURCE_ERASER:
            # enter eraser mode
            adj.set_value(1.0)
        elif new_device.source != gdk.SOURCE_ERASER and \
               (old_device is None or old_device.source == gdk.SOURCE_ERASER):
            # leave eraser mode
            adj.set_value(0.0)
        print 'device change:', new_device.name

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
        
    def with_wait_cursor(func):
        """python decorator that adds a wait cursor around a function"""
        def wrapper(self, *args, **kwargs):
            self.window.set_cursor(gdk.Cursor(gdk.WATCH))
            self.tdw.window.set_cursor(None)
            # make sure it is actually changed before we return
            while gtk.events_pending():
                gtk.main_iteration(False)
            try:
                func(self, *args, **kwargs)
            finally:
                self.window.set_cursor(None)
                self.tdw.update_cursor(force=True)
        return wrapper
    
    @with_wait_cursor
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
            self.filename = os.path.abspath(filename)
            print 'Loaded from', self.filename
            self.reset_view_cb(None)
            self.tdw.recenter_document()

    @with_wait_cursor
    def save_file(self, filename, **options):
        try:
            x, y, w, h =  self.doc.get_bbox()
            assert w > 0 and h > 0, 'The canvas is empty.'
            self.doc.save(filename, **options)
        except Exception, e:
            print 'Failed to save, traceback:'
            traceback.print_exc()
            d = gtk.MessageDialog(self, type=gtk.MESSAGE_ERROR, buttons=gtk.BUTTONS_OK)
            d.set_markup('Failed to save:\n' + str(e))
            d.run()
            d.destroy()
        else:
            self.filename = os.path.abspath(filename)
            print 'Saved to', self.filename
            self.save_history.append(os.path.abspath(filename))
            self.save_history = self.save_history[-100:]
            f = open(os.path.join(self.app.confpath, 'save_history.conf'), 'w')
            f.write('\n'.join(self.save_history))
            ## tell other gtk applications
            ## (hm, is there any application that uses this at all? or is the code below wrong?)
            #manager = gtk.recent_manager_get_default()
            #manager.add_item(os.path.abspath(filename))

    def confirm_destructive_action(self, title='Confirm', question='Really continue?'):
        t = self.doc.unsaved_painting_time
        if t < 30:
            # no need to ask
            return True

        if t > 120:
            t = '%d minutes' % (t/60)
        else:
            t = '%d seconds' % t
        d = gtk.Dialog(title, 
                       self,
                       gtk.DIALOG_MODAL,
                       (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                        gtk.STOCK_DISCARD, gtk.RESPONSE_OK))
        d.set_has_separator(False)
        d.set_default_response(gtk.RESPONSE_OK)
        l = gtk.Label()
        l.set_markup("<b>" + question + "</b>\n\nThis will discard %s of unsaved painting." % t)
        l.set_padding(10, 10)
        l.show()
        d.vbox.pack_start(l)
        response = d.run()
        d.destroy()
        return response == gtk.RESPONSE_OK

    def new_cb(self, action):
        if not self.confirm_destructive_action():
            return
        bg = self.doc.background
        self.doc.clear()
        self.doc.set_background(bg)
        self.filename = None

    def open_cb(self, action):
        if not self.confirm_destructive_action():
            return
        dialog = gtk.FileChooserDialog("Open..", self,
                                       gtk.FILE_CHOOSER_ACTION_OPEN,
                                       (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                        gtk.STOCK_OPEN, gtk.RESPONSE_OK))
        dialog.set_default_response(gtk.RESPONSE_OK)

        f = gtk.FileFilter()
        f.set_name("All Recognized Formats")
        f.add_pattern("*.ora")
        f.add_pattern("*.png")
        f.add_pattern("*.jpg")
        f.add_pattern("*.jpeg")
        dialog.add_filter(f)

        f = gtk.FileFilter()
        f.set_name("OpenRaster (*.ora)")
        f.add_pattern("*.ora")
        dialog.add_filter(f)

        f = gtk.FileFilter()
        f.set_name("PNG (*.png)")
        f.add_pattern("*.png")
        dialog.add_filter(f)

        f = gtk.FileFilter()
        f.set_name("JPEG (*.jpg; *.jpeg)")
        f.add_pattern("*.jpg")
        f.add_pattern("*.jpeg")
        dialog.add_filter(f)

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


    def init_save_dialog(self):
        dialog = gtk.FileChooserDialog("Save..", self,
                                       gtk.FILE_CHOOSER_ACTION_SAVE,
                                       (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                        gtk.STOCK_SAVE, gtk.RESPONSE_OK))
        self.save_dialog = dialog
        dialog.set_default_response(gtk.RESPONSE_OK)
        dialog.set_do_overwrite_confirmation(True)

        filter2info = {}
        self.filter2info = filter2info

        f = gtk.FileFilter()
        filter2info[f] = ('.ora', {})
        f.set_name("Any format (prefer OpenRaster)")
        self.save_filter_default = f

        f.add_pattern("*.png")
        f.add_pattern("*.ora")
        f.add_pattern("*.jpg")
        f.add_pattern("*.jpeg")
        dialog.add_filter(f)

        f = gtk.FileFilter()
        filter2info[f] = ('.ora', {})
        f.set_name("OpenRaster (*.ora)")
        f.add_pattern("*.ora")
        dialog.add_filter(f)

        f = gtk.FileFilter()
        filter2info[f] = ('.png', {'alpha': False})
        f.set_name("PNG solid with background (*.png)")
        f.add_pattern("*.png")
        dialog.add_filter(f)

        f = gtk.FileFilter()
        filter2info[f] = ('.png', {'alpha': True})
        f.set_name("PNG transparent (*.png)")
        f.add_pattern("*.png")
        dialog.add_filter(f)

        f = gtk.FileFilter()
        filter2info[f] = ('.jpg', {'quality': 90})
        f.set_name("JPEG 90% quality (*.jpg; *.jpeg)")
        f.add_pattern("*.jpg")
        f.add_pattern("*.jpeg")
        dialog.add_filter(f)

    def save_as_cb(self, action):
        dialog = self.save_dialog

        def dialog_set_filename(s):
            # According to pygtk docu we should use set_filename(),
            # however doing so removes the selected filefilter.
            path, name = os.path.split(s)
            dialog.set_current_folder(path)
            dialog.set_current_name(name)

        if self.filename:
            dialog_set_filename(self.filename)
        else:
            dialog_set_filename('')
            dialog.set_filter(self.save_filter_default)

        try:
            while dialog.run() == gtk.RESPONSE_OK:

                filename = dialog.get_filename()
                name, ext = os.path.splitext(filename)
                ext_filter, options = self.filter2info.get(dialog.get_filter(), ('ora', {}))

                if ext:
                    if ext_filter != ext:
                        # Minor ugliness: if the user types '.png' but
                        # leaves the default .ora filter selected, we
                        # use the default options instead of those
                        # above. However, they are the same at the moment.
                        options = {}
                    assert(filename)
                    self.save_file(filename, **options)
                    break
                
                # add proper extension
                filename = name + ext_filter

                # trigger overwrite confirmation for the modified filename
                dialog_set_filename(filename)
                dialog.response(gtk.RESPONSE_OK)

        finally:
            dialog.hide()

    def save_scrap_cb(self, action):
        filename = self.filename
        prefix = self.app.settingsWindow.save_scrap_prefix

        number = None
        if filename:
            l = re.findall(re.escape(prefix) + '([0-9]+)', filename)
            if l:
                number = l[0]

        if number:
            # reuse the number, find the next character
            char = 'a'
            for filename in glob(prefix + number + '_*'):
                c = filename[len(prefix + number + '_')]
                if c >= 'a' and c <= 'z' and c >= char:
                    char = chr(ord(c)+1)
            if char > 'z':
                # out of characters, increase the number
                self.filename = None
                return self.save_scrap_cb(action)
            filename = '%s%s_%c' % (prefix, number, char)
        else:
            # we don't have a scrap filename yet, find the next number
            maximum = 0
            for filename in glob(prefix + '[0-9][0-9][0-9]*'):
                filename = filename[len(prefix):]
                res = re.findall(r'[0-9]*', filename)
                if not res: continue
                number = int(res[0])
                if number > maximum:
                    maximum = number
            filename = '%s%03d_a' % (prefix, maximum+1)

        #if self.doc.is_layered():
        #    filename += '.ora'
        #else:
        #    filename += '.png'
        filename += '.ora'

        assert not os.path.exists(filename)
        self.save_file(filename)

    def open_recent_cb(self, action):
        # feed history with scrap directory (mainly for initial history)
        prefix = self.app.settingsWindow.save_scrap_prefix
        prefix = os.path.abspath(prefix)
        l = glob(prefix + '*.png') + glob(prefix + '*.ora') + glob(prefix + '*.jpg') + glob(prefix + '*.jpeg')
        l = [x for x in l if x not in self.save_history]
        l = l + self.save_history
        l = [x for x in l if os.path.exists(x)]
        l.sort(key=os.path.getmtime)
        self.save_history = l

        # pick the next most recent file from the history
        idx = -1
        if self.filename in self.save_history:
            def basename(filename):
                return os.path.splitext(filename)[0]
            idx = self.save_history.index(self.filename)
            while basename(self.save_history[idx]) == basename(self.filename):
                idx -= 1
                if idx == -1:
                    return

        if not self.confirm_destructive_action():
            return
        self.open_file(self.save_history[idx])

    def quit_cb(self, *trash):
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
        else: assert 0
        if self.zoomlevel < 0: self.zoomlevel = 0
        if self.zoomlevel >= len(self.zoomlevel_values): self.zoomlevel = len(self.zoomlevel_values) - 1
        z = self.zoomlevel_values[self.zoomlevel]
        self.tdw.set_zoom(z)

    def rotate(self, command):
        if   command == 'RotateRight': self.tdw.rotate(+2*math.pi/14)
        elif command == 'RotateLeft' : self.tdw.rotate(-2*math.pi/14)
        else: assert 0

    def reset_view_cb(self, command):
        self.tdw.set_rotation(0.0)
        self.zoomlevel = self.zoomlevel_values.index(1.0)
        self.tdw.set_zoom(1.0)
        self.tdw.set_flipped(False)
        self.action_group.get_action('Flip').set_active(False)

    def fullscreen_cb(self, *trash):
        # note: there is some ugly flickering when toggling fullscreen
        #       self.window.begin_paint/end_paint does not help against it
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            x, y = self.get_position()
            w, h = self.get_size()
            self.geometry_before_fullscreen = (x, y, w, h)
            self.menubar.hide()
            self.window.fullscreen()
            self.tdw.set_scroll_at_edges(True)
        else:
            self.window.unfullscreen()
            self.menubar.show()
            self.tdw.set_scroll_at_edges(False)
            del self.geometry_before_fullscreen

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
            u"Copyright (C) 2005-2009\n"
            u"Martin Renold &lt;martinxyz@gmx.ch&gt;\n\n"
            u"Contributors:\n"
            u"Artis Rozentāls &lt;artis@aaa.apollo.lv&gt; (brushes)\n"
            u"Yves Combe &lt;yves@ycombe.net&gt; (portability)\n"
            u"Sebastian Kraft (desktop icon)\n"
            u"Popolon &lt;popolon@popolon.org&gt; (brushes)\n"
            u"Clement Skau &lt;clementskau@gmail.com&gt; (programming)\n"
            u'Marcelo "Tanda" Cerviño &lt;info@lodetanda.com.ar&gt; (patterns, brushes)\n'
            u'Jon Nordby &lt;jononor@gmail.com&gt; (programming)\n'
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
        d.set_markup("There is a tutorial available "
                     "on the MyPaint homepage. It explains some features which are "
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
            "or with the arrow keys."
            "\n\n"
            "In contrast to earlier versions, scrolling and zooming are harmless now and "
            "will not make you run out of memory. But you still require a lot of memory "
            "if you paint all over while fully zoomed out."
            )
        d.run()
        d.destroy()

