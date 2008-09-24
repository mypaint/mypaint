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
MYPAINT_VERSION="0.5.1"
import gtk, os, zlib, random, re, math
from gtk import gdk, keysyms
import tileddrawwidget
import tilelib, brush, document
import command
from time import time
from glob import glob

class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app

        self.set_title('MyPaint')
        def delete_event_cb(window, event, app): return app.quit()
        self.connect('delete-event', delete_event_cb, self.app)
        self.connect('key-press-event', self.key_press_event_cb_before)
        self.connect('key-release-event', self.key_release_event_cb_before)
        self.connect_after('key-press-event', self.key_press_event_cb_after)
        self.connect_after('key-release-event', self.key_release_event_cb_after)
        self.connect("button-press-event", self.button_press_cb)
        self.connect("button-release-event", self.button_release_cb)
        self.connect("scroll-event", self.scroll_cb)
        self.set_size_request(600, 400)
        vbox = gtk.VBox()
        self.add(vbox)

        self.create_ui()
        self.menubar = self.ui.get_widget('/Menubar')
        vbox.pack_start(self.menubar, expand=False)

        self.tdw = tileddrawwidget.TiledDrawWidget()
        self.tdw.set_brush(self.app.brush)
        vbox.pack_start(self.tdw)
        self.tdw.toolchange_observers.append(self.toolchange_cb)
        #self.tdw.connect("gesture-recognized", self.gesture_recognized_cb)

        self.statusbar = sb = gtk.Statusbar()
        vbox.pack_end(sb, expand=False)

        self.layer = document.Layer()
        self.layers = [self.layer]
        self.show_layers_above = False
        self.update_layers()

        #self.zoomlevel_values = [0.09, 0.12,  0.18, 0.25, 0.33,  0.50, 0.66,  1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 8.0]
        self.zoomlevel_values = [      0.12, 2.0/11, 0.25, 1.0/3, 0.50, 2.0/3, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 8.0, 16.0]
        self.zoomlevel = self.zoomlevel_values.index(1.0)
        self.fullscreen = False

        self.modifying = False
        self.paint_below_stroke = None

        self.command_stack = command.CommandStack()
        self.stroke = document.Stroke()
        self.stroke.start_recording(self.tdw, self.app.brush)
        self.pending_actions = [self.split_stroke]
        self.app.brush.observers.append(self.brush_modified_cb)
        self.app.brush.set_split_stroke_callback(self.split_stroke_cb)

        self.last_gesture_time = 0
        
    def create_ui(self):
        ag = gtk.ActionGroup('WindowActions')
        # FIXME: this xml menu only creates unneeded information duplication, I think.
		# FIXME: better just use glade...
        ui_string = """<ui>
          <menubar name='Menubar'>
            <menu action='FileMenu'>
              <menuitem action='Clear'/>
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
              <menuitem action='ModifyLastStroke'/>
              <menuitem action='ModifyEnd'/>
              <separator/>
              <menuitem action='LowerLastStroke'/>
              <menuitem action='RaiseLastStroke'/>
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
              <separator/>
              <menuitem action='MoveLeft'/>
              <menuitem action='MoveRight'/>
              <menuitem action='MoveUp'/>
              <menuitem action='MoveDown'/>
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
              <menuitem action='InvertColor'/>
              <menuitem action='PickColor'/>
              <menuitem action='ChangeColor'/>
              <menuitem action='ColorSelectionWindow'/>
            </menu>
            <menu action='LayerMenu'>
              <menuitem action='LayerBG'/>
              <menuitem action='LayerFG'/>
              <menuitem action='NewLayer'/>
              <menuitem action='ToggleAbove'/>
            </menu>
            <menu action='DebugMenu'>
              <menuitem action='PrintInputs'/>
              <menuitem action='DisableGammaCorrection'/>
              <menuitem action='Test'/>
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
            ('Clear',        None, 'Clear', '<control>period', None, self.clear_cb),
            ('Open',         None, 'Open...', '<control>O', None, self.open_cb),
            ('Save',         None, 'Save', '<control>S', None, self.save_cb),
            ('SaveAs',       None, 'Save As...', '<control><shift>S', None, self.save_as_cb),
            ('SaveNext',     None, 'Save Next', 'F2', None, self.save_next_cb),
            ('Quit',         None, 'Quit', None, None, self.quit_cb),


            ('EditMenu',           None, 'Edit'),
            ('Undo',               None, 'Undo', '<control>Z', None, self.undo_cb),
            ('Redo',               None, 'Redo', '<control>Y', None, self.redo_cb),
            ('ModifyLastStroke',   None, 'Modify Last Stroke', 'm', None, self.modify_last_stroke_cb),
            ('ModifyEnd',          None, 'Stop Modifying', 'n', None, self.modify_end_cb),
            ('LowerLastStroke',    None, 'Lower Last Stroke (Experimental, slow!)', 'Page_Down', None, self.lower_or_raise_last_stroke_cb),
            ('RaiseLastStroke',    None, 'Raise Last Stroke', 'Page_Up', None, self.lower_or_raise_last_stroke_cb),

            ('BrushMenu',    None, 'Brush'),
            ('InvertColor',  None, 'Invert Color', 'x', None, self.invert_color_cb),
            ('Brighter',     None, 'Brighter', None, None, self.brighter_cb),
            ('Darker',       None, 'Darker', None, None, self.darker_cb),
            ('Bigger',       None, 'Bigger', 'f', None, self.brush_bigger_cb),
            ('Smaller',      None, 'Smaller', 'd', None, self.brush_smaller_cb),
            ('MoreOpaque',   None, 'More Opaque', None, None, self.more_opaque_cb),
            ('LessOpaque',   None, 'Less Opaque', None, None, self.less_opaque_cb),
            ('PickColor',    None, 'Pick Color', 'r', None, self.pick_color_cb),
            ('ChangeColor',  None, 'Change Color', 'v', None, self.change_color_cb),

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
            ('LayerBG',      None, 'Background (take layer away)', None, None, self.layer_bg_cb),
            ('LayerFG',      None, 'Foreground (put layer back)',  None, None, self.layer_fg_cb),
            ('NewLayer',     None, 'New Layer', None, None, self.new_layer_cb),
            ('ToggleAbove',  None, 'Toggle Layers Above Current', 'a', None, self.toggle_layers_above_cb),

            ('DialogMenu',  None, 'Dialogs'),
            ('BrushSelectionWindow',  None, 'Brush List', 'b', None, self.toggleBrushSelectionWindow_cb),
            ('BrushSettingsWindow',   None, 'Brush Settings', '<control>b', None, self.toggleBrushSettingsWindow_cb),
            ('ColorSelectionWindow',  None, 'GTK Color Dialog', 'g', None, self.toggleColorSelectionWindow_cb),
            ('SettingsWindow',        None, 'Settings', None, None, self.toggleSettingsWindow_cb),

            ('HelpMenu',     None, 'Help'),
            ('Docu', None, 'Where is the Documentation?', None, None, self.show_docu_cb),
            ('ShortcutHelp',  None, 'Change the Keyboard Shortcuts?', None, None, self.shortcut_help_cb),
            ('About', None, 'About MyPaint', None, None, self.show_about_cb),

            ('DebugMenu',    None, 'Debug'),
            ('Test', None, 'Test', None, None, self.test_cb),


            ('ShortcutsMenu', None, 'Shortcuts'),

            ('ViewMenu', None, 'View'),
            ('Fullscreen',   None, 'Fullscreen', 'F11', None, self.fullscreen_cb),
            ('ZoomIn',       None, 'Zoom In', 'plus', None, self.zoom_cb),
            ('ZoomOut',      None, 'Zoom Out', 'minus', None, self.zoom_cb),
            ('Zoom1',        None, 'Zoom 1:1', 'z', None, self.zoom_cb),
            ('RotateRight',  None, 'Rotate Clockwise', 'comma', None, self.rotate_cb),
            ('RotateLeft',   None, 'Rotate Counterclockwise', 'period', None, self.rotate_cb),
            ('Rotate0',      None, 'Rotate Upright', None, None, self.rotate_cb),
            ('MoveLeft',     None, 'Look Left', None, None, self.move_cb),
            ('MoveRight',    None, 'Look Right', None, None, self.move_cb),
            ('MoveUp',       None, 'Look Up', None, None, self.move_cb),
            ('MoveDown',     None, 'Look Down', None, None, self.move_cb),
            ('ViewHelp',     None, 'Help', None, None, self.view_help_cb),
            ]
        ag.add_actions(actions)
        toggle_actions = [
            # name, stock id, label, accelerator, tooltip, callback, default toggle status
            ('PrintInputs', None, 'Print Brush Input Values to stdout', None, None, self.print_inputs_cb),
            ('DisableGammaCorrection', None, 'Disable sRGB Gamma Correction', None, None, self.disableGammaCorrection_cb),
            ]
        ag.add_toggle_actions(toggle_actions)
        self.ui = gtk.UIManager()
        self.ui.insert_action_group(ag, 0)
        self.ui.add_ui_from_string(ui_string)
        self.app.accel_group = self.ui.get_accel_group()
        self.add_accel_group(self.app.accel_group)

    def toggleWindow(self, w):
        # TODO: make some windows "real" popups at mouse pos when invoked via keystrokes?
        if w.is_active():
            w.hide()
        else:
            w.show_all() # might be for the first time
            w.present()
    def toggleBrushSelectionWindow_cb(self, action):
        self.toggleWindow(self.app.brushSelectionWindow)
    def toggleBrushSettingsWindow_cb(self, action):
        self.toggleWindow(self.app.brushSettingsWindow)
    def toggleColorSelectionWindow_cb(self, action):
        self.toggleWindow(self.app.colorSelectionWindow)
    def toggleSettingsWindow_cb(self, action):
        self.toggleWindow(self.app.settingsWindow)

    def print_inputs_cb(self, action):
        self.app.brush.print_inputs = action.get_active()

    def test_cb(self, action):
        self.tdw.layer.save('test.png')

    def disableGammaCorrection_cb(self, action):
		self.tdw.disableGammaCorrection = action.get_active()
		self.tdw.queue_draw()

    def finish_pending_actions(self, skip=None):
        # this function must be called before manipulation the command stack
        for f in self.pending_actions[:]:
            if f == skip: continue
            if f not in self.pending_actions: continue # list modified
            f()

    def split_stroke(self):
        # let the brush emit the signal (this calls self.split_stroke_cb)
        self.app.brush.split_stroke()

    def split_stroke_cb(self):
        self.stroke.stop_recording()
        if not self.stroke.empty:
            self.finish_pending_actions(skip=self.split_stroke)
            pbs = self.paint_below_stroke
            if pbs and pbs in self.layer.strokes:
                z = self.layer.strokes.index(pbs)
            else:
                z = -1

            self.command_stack.add(command.Stroke(self.layer, self.stroke, z))
            self.layer.rerender()
            self.layer.populate_cache()

            # remove "saved to..." etc.
            self.statusbar.pop(1)

        self.stroke = document.Stroke()
        self.stroke.start_recording(self.tdw, self.app.brush)

    def undo_cb(self, action):
        self.finish_pending_actions()
        self.command_stack.undo()

        cost = self.layer.rerender(only_estimate_cost=True)
        if cost > 50:
            d = gtk.MessageDialog(
                 type = gtk.MESSAGE_QUESTION,
                 flags = gtk.DIALOG_MODAL,
                 buttons = gtk.BUTTONS_YES_NO,
                 message_format="This undo step will require %d brush strokes to be re-rendered. This might take some time.\n\nDo you really want to undo?" % cost
                 )
            if d.run() != gtk.RESPONSE_YES:
                self.command_stack.redo()
            d.destroy()

        self.layer.rerender()

    def redo_cb(self, action):
        self.finish_pending_actions()
        self.command_stack.redo()
        self.layer.rerender()

    def get_recent_strokes(self, max_count):
        assert max_count > 0
        result = self.layer.strokes[:] # copy
        def cmpfunc(a, b):
            return cmp(a.serial_number, b.serial_number)
        result.sort(cmpfunc, reverse=True)
        return result[:max_count]

    def modify_last_stroke_cb(self, action):
        self.start_or_continue_modifying()

    def start_or_continue_modifying(self, count=1):
        self.finish_pending_actions(skip=self.end_modifying)
        if self.modifying:
            assert self.end_modifying in self.pending_actions
            count = self.modifying + 1
            if count > len(self.layer.strokes):
                print 'All strokes selected already!'
                return
            self.command_stack.undo()
        else:
            assert self.end_modifying not in self.pending_actions
            self.pending_actions.append(self.end_modifying)

        strokes = self.get_recent_strokes(count)
        cmd = command.ModifyStrokes(self.layer, strokes, self.app.brush)
        self.command_stack.add(cmd)

        self.layer.rerender()

        if not self.modifying:
            self.statusbar.push(3, 'modifying - change brush or color now')
        else:
            self.statusbar.pop(3)
            self.statusbar.push(3, 'modifying %d strokes' % count)

        self.modifying = count

        self.last_modifying_time = time()

    def end_modifying(self):
        assert self.modifying
        self.statusbar.pop(3)
        self.modifying = False
        self.pending_actions.remove(self.end_modifying)

    def modify_end_cb(self, action):
        if self.modifying:
            self.end_modifying()

    def lower_or_raise_last_stroke_cb(self, action):
        self.finish_pending_actions()
        action = action.get_name()

        cmd = self.command_stack.get_last_command()
        if not isinstance(cmd, command.Stroke):
            self.statusbar.push(4, 'last command was not a stroke')
            return

        # note: you can undo->lower->redo
        # this history manipluation is not in the undo/redo spirit
        # let's say it's a feature, not a bug
        self.command_stack.undo()

        if action == 'LowerLastStroke':
            # lower it such that the visible result changes
            intersections = []
            rect = cmd.stroke.bbox
            for z, stroke in enumerate(self.layer.strokes):
                stroke.z = z
                if stroke is cmd.stroke: continue
                if rect.overlaps(stroke.bbox):
                    intersections.append(stroke)

            below = [stroke for stroke in intersections if stroke.z < cmd.z]
            print len(below), 'strokes are below'
            if below:
                def cmpfunc(a, b):
                    return cmp(a.z, b.z)
                below.sort(cmpfunc)
                cmd.z = below[-1].z
            else:
                cmd.z = 0

            # clean up
            for stroke in self.layer.strokes:
                del stroke.z

        elif action == 'RaiseLastStroke':
            # raise to top, because it is cheapest to render there
            cmd.z = len(self.layer.strokes)
        else:
            assert False
                
        if cmd.z < 0:
            cmd.z = 0
        if cmd.z >= len(self.layer.strokes):
            cmd.z = len(self.layer.strokes)
            self.paint_below_stroke = None
        else:
            self.paint_below_stroke = self.layer.strokes[cmd.z]

        self.command_stack.redo()
        self.layer.rerender()

        if self.paint_below_stroke:
            self.statusbar.push(4, 'painting below other strokes (slow)')
        else:
            self.statusbar.push(4, '')

    def brush_modified_cb(self):
        # called at every brush setting modification, should return fast
        self.split_stroke()

        if self.modifying:
            self.finish_pending_actions(skip=self.end_modifying)
            cmd = self.command_stack.get_last_command()
            if isinstance(cmd, command.ModifyStrokes):
                count = self.modifying

                if time() - self.last_modifying_time > 3:
                    # split into a different undo action
                    # (so the user can recover the previous modification if he selected a brush and forgot that he was still modifying)
                    self.end_modifying()
                    self.start_or_continue_modifying(count)
                    cmd = self.command_stack.get_last_command()
                self.last_modifying_time = time()

                #print 'redo', count, 'modified strokes'
                self.command_stack.undo()
                cmd.set_new_brush(self.app.brush)
                self.command_stack.add(cmd)
                self.layer.rerender()

                if count == 1:
                    self.statusbar.pop(3)
                    self.statusbar.push(3, 'modifying one stroke (hit again to add more)')

    def toolchange_cb(self):
        # FIXME: add argument with tool id, and remember settings
        # also make sure proximity events outside the window are checked
        self.split_stroke()

    def key_press_event_cb_before(self, win, event):
        key = event.keyval 
        ANY_MODIFIER = gdk.SHIFT_MASK | gdk.MOD1_MASK | gdk.CONTROL_MASK
        if event.state & ANY_MODIFIER:
            # allow user shortcuts with modifiers
            return False
        if key == keysyms.Left: self.move('MoveLeft')
        elif key == keysyms.Right: self.move('MoveRight')
        elif key == keysyms.Up: self.move('MoveUp')
        elif key == keysyms.Down: self.move('MoveDown')
        elif key == keysyms.space: self.tdw.dragging = True
        else: return False
        return True
    def key_release_event_cb_before(self, win, event):
        if event.keyval == keysyms.space:
            self.tdw.dragging = False
            return True
        return False

    def key_press_event_cb_after(self, win, event):
        # Not checking modifiers because this function gets only 
        # called if no user keybinding accepted the event.
        if event.keyval == keysyms.KP_Add: self.zoom('ZoomIn')
        elif event.keyval == keysyms.KP_Subtract: self.zoom('ZoomOut')
        elif self.fullscreen and event.keyval == keysyms.Escape: self.fullscreen_cb()
        else: return False
        return True
    def key_release_event_cb_after(self, win, event):
        return False


    def button_press_cb(self, win, event):
        #print event.device, event.button
        if event.button == 2:
            self.tdw.dragging = True

    def button_release_cb(self, win, event):
        #print event.device, event.button
        if event.button == 2:
            self.tdw.dragging = False

    def scroll_cb(self, win, event):
        d = event.direction
        if event.state & gdk.CONTROL_MASK:
            if d == gdk.SCROLL_UP:
                self.zoom('ZoomIn')
            elif d == gdk.SCROLL_DOWN:
                self.zoom('ZoomOut')
        else:
            self.move({
                gdk.SCROLL_RIGHT: 'MoveRight',
                gdk.SCROLL_LEFT: 'MoveLeft',
                gdk.SCROLL_UP: 'MoveUp',
                gdk.SCROLL_DOWN: 'MoveDown',
                }[d])

    def clear_cb(self, action):
        self.finish_pending_actions()
        cmd = command.ClearLayer(self.layer)
        self.command_stack.add(cmd)
        self.statusbar.pop(1)
        self.filename = None
        self.layer.rerender()
        
    def update_layers(self):
        self.tdw.layer = self.layer.surface
        l = []
        for layer in self.layers:
            l.append(layer.surface)
            if not self.show_layers_above and layer is self.layer:
                break
        self.tdw.displayed_layers = l
        self.tdw.queue_draw()

    def layer_bg_cb(self, action):
        # TODO: make an action to allow undo
        self.split_stroke()
        i = self.layers.index(self.layer)
        i -= 1
        if i < 0: return
        self.layer = self.layers[i]
        self.update_layers()

    def layer_fg_cb(self, action):
        # TODO: make an action to allow undo
        self.split_stroke()
        i = self.layers.index(self.layer)
        i += 1
        if i >= len(self.layers): return
        self.layer = self.layers[i]
        self.update_layers()

    def new_layer_cb(self, action):
        # TODO: make an action to allow undo
        self.split_stroke()
        i = self.layers.index(self.layer)
        self.layer = document.Layer()
        self.layers.insert(i+1, self.layer)
        self.update_layers()

    def toggle_layers_above_cb(self, action):
        self.show_layers_above = not self.show_layers_above
        self.update_layers()

    def invert_color_cb(self, action):
        self.app.brush.invert_color()
        
    def pick_color_cb(self, action):
        self.app.colorSelectionWindow.pick_color_at_pointer()

    def change_color_cb(self, action):
        self.app.colorSelectionWindow.show_change_color_window()

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
        self.finish_pending_actions()
        self.statusbar.pop(1)
        try:
            pixbuf = gdk.pixbuf_new_from_file(filename)
            cmd = command.LoadImage(self.layer, pixbuf)
            self.command_stack.add(cmd)
            self.layer.rerender()
        except Exception, e:
            d = gtk.MessageDialog(self, type=gtk.MESSAGE_ERROR, buttons=gtk.BUTTONS_OK)
            d.set_markup(str(e))
            d.run()
            d.destroy()
            print e
            self.clear_cb(None)
        else:
            self.statusbar.push(1, 'Loaded from ' + filename)
            self.filename = filename

    def save_file(self, filename):
        self.finish_pending_actions()
        self.filename = filename
        self.statusbar.pop(1)
        try:
            self.tdw.save(filename)
        except Exception, e:
            print e
            d = gtk.MessageDialog(self, type=gtk.MESSAGE_ERROR, buttons=gtk.BUTTONS_OK)
            d.set_markup(str(e))
            d.run()
            d.destroy()
            self.statusbar.push(1, 'Failed to save!')
        else:
            self.statusbar.push(1, 'Saved to ' + filename)

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

        if self.filename:
            dialog.set_filename(self.filename)
        if dialog.run() == gtk.RESPONSE_OK:
            self.open_file(dialog.get_filename())
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
        filter = gtk.FileFilter()
        filter.set_name("png")
        filter.add_pattern("*.png")
        dialog.add_filter(filter)

        if self.filename:
            dialog.set_filename(self.filename)
        if dialog.run() == gtk.RESPONSE_OK:
            filename = dialog.get_filename()
            trash, ext = os.path.splitext(filename)
            if not ext:
                filename += '.png'
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

    def quit_cb(self, action):
        self.finish_pending_actions()
        return self.app.quit()

    def move_cb(self, action):
        self.move(action.get_name())
    def zoom_cb(self, action):
        self.zoom(action.get_name())
    def rotate_cb(self, action):
        self.rotate(action.get_name())

    def move(self, command):
        self.split_stroke()
        step = min(self.tdw.window.get_size()) / 5
        if command == 'MoveLeft':
            self.tdw.scroll(-step, 0)
        elif command == 'MoveRight':
            self.tdw.scroll(+step, 0)
        elif command == 'MoveUp':
            self.tdw.scroll(0, -step)
        elif command == 'MoveDown':
            self.tdw.scroll(0, +step)
        else:
            assert 0

    def zoom(self, command):
        if command == 'ZoomIn':
            self.zoomlevel += 1
        elif command == 'ZoomOut':
            self.zoomlevel -= 1
        elif command == 'Zoom1':
            self.zoomlevel = self.zoomlevel_values.index(1.0)
        else:
            assert 0
        if self.zoomlevel < 0: self.zoomlevel = 0
        if self.zoomlevel >= len(self.zoomlevel_values): self.zoomlevel = len(self.zoomlevel_values) - 1
        z = self.zoomlevel_values[self.zoomlevel]
        #self.statusbar.push(2, 'Zoom %.2f' % z)
        #print 'Zoom %.2f' % z

        self.split_stroke()
        self.tdw.set_zoom(z)

    def rotate(self, command):
        self.split_stroke()
        if command == 'RotateRight':
            self.tdw.rotate(+2*math.pi/14)
        elif command == 'RotateLeft':
            self.tdw.rotate(-2*math.pi/14)
        elif command == 'Rotate0':
            self.tdw.set_rotation(0.0)
        else:
            assert 0

    def fullscreen_cb(self, *trash):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.statusbar.hide()
            self.menubar.hide()
            self.window.fullscreen()
        else:
            self.window.unfullscreen()
            self.menubar.show()
            self.statusbar.show()

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

    def gesture_recognized_cb(self, widget):
        print 'pick-color gesture recognized'
        t = time()
        if t - self.last_gesture_time < 1.0:
            # double-click
            self.app.colorSelectionWindow.show_change_color_window()
        else:
            self.app.colorSelectionWindow.pick_color_at_pointer()
        self.last_gesture_time = t

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
            u"\n"
            u"This program is free software; you can redistribute it and/or modify "
            u"it under the terms of the GNU General Public License as published by "
            u"the Free Software Foundation; either version 2 of the License, or "
            u"(at your option) any later version.\n"
            u"\n"
            u"This program is distributed in the hope that it will be useful,"
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
            "You can also drag the canvas with the middle mouse button or "
            "with the arrow keys.\n\n"
            "In contrast to earlier versions, scrolling and zooming are harmless now and "
            "will not make you run out of memory. But you still require a lot of memory "
            "if you paint all over while fully zoomed out."
            )
        d.run()
        d.destroy()

