# -*- coding: utf-8 -*-
#
# This file is part of MyPaint.
# Copyright (C) 2016 by Grzegorz Wójcik <grzegorz.w.1597@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"gui part of perspecive mode"

# Imports
import os.path
import weakref
from gettext import gettext as _
import math
import logging

from gi.repository import Gtk
from gi.repository import Gio
from gi.repository import Gdk
from gi.repository import GdkPixbuf
from gi.repository import GLib
from gi.repository import PangoCairo

import gui.mode
import gui.dialogs

from gui.overlays import rounded_box
from gui.perspectivedata import (get_default_data, update_data)

from lib.color import (RGBColor)
import lib.libperspective as p
from lib.libperspective import (
    Quaternion, get_line_bounding_box, teselate_poligon,
    intersect_view_ray_and_plane, PythonGraph
)

logger = logging.getLogger(__name__)


# Class defs


class State(object):
    """Enumeration of the states that an PerspectiveMode can be in"""
    NOOP = 0
    DRAW = 1
    ADD_POINT = 2
    DRAG_POINT = 4  # TODO rm ?

    def __init__(self):
        self._state = self.NOOP
        self._tool = 'LINE'
        self._generator = None
        self.polygon_side_count = 3

    def set_state(self, new_state):
        "Set state"
        if new_state == self.NOOP and self._generator is not None:
            new_state = next(self._generator, None)
            if new_state is None:
                self._generator = None
                new_state = self.NOOP
        self._state = new_state

    # TODO unused
    def set_generator(self, generator):
        "Set generator for automatic state changes"
        self._generator = generator

    def is_draw(self):
        "is in draw state"
        return self._state == self.DRAW

    def is_drag(self):
        "is in point dragging state"
        return self._state == self.DRAG_POINT

    def is_add(self):
        "is in point adding state"
        return bool(self._state & self.ADD_POINT)

    def is_noop(self):
        "is in idle state"
        return self._state == self.NOOP

    def is_line_tool(self):
        "is line tool selected"
        return self._tool == 'LINE'

    def is_circle_tool(self):
        "is circle tool selected"
        return self._tool == 'CIRCLE'

    def is_polygon_tool(self):
        "is poligon tool selected"
        return self._tool == 'POLYGON'

    def set_tool_line(self):
        "use line tool"
        self._tool = 'LINE'

    def set_tool_circle(self):
        "use circle tool"
        self._tool = 'CIRCLE'

    def set_tool_polygon(self):
        "use polygon tool"
        self._tool = 'POLYGON'


class OverlayWidgets(object):
    def __init__(self):
        self._widgets = []
        self._last_pressed = None

    def add(self, widget):
        self._widgets.append(widget)

    def clear(self):
        self._widgets = []
        self._last_pressed = None

    def getAll(self):
        return self._widgets

    def button_press(self, model_pos):
        for widget in self._widgets:
            is_pressed = widget.test_button_press(model_pos)
            if is_pressed:
                self._last_pressed = widget
                return True
        self._last_pressed = None
        return False

    def is_drag(self):
        if self._last_pressed:
            return True
        return False

    def drag(self, graph, option_presenter, model_pos):
        self._last_pressed.drag(graph, model_pos)
        for node in self._last_pressed.drag_nodes_to_update():
            option_presenter.update_description(node.uid)

    def drag_stop(self):
        # TODO
        pass


class PerspectiveMode(gui.mode.ScrollableModeMixin,
                      gui.mode.BrushworkModeMixin,
                      gui.mode.DragMode):
    "Perspective mode class"

    # Metadata properties

    ACTION_NAME = "PerspectiveMode"
    pointer_behavior = gui.mode.Behavior.PAINT_FREEHAND
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW
    permitted_switch_actions = (
        set(gui.mode.BUTTON_BINDING_ACTIONS).union([
            'RotateViewMode',
            'ZoomViewMode',
            'PanViewMode',
        ])
    )

    _OPTIONS_PRESENTER = None   #: Options presenter singleton

    _graph = PythonGraph()
    _state = State()

    _SETTINGS_KEY = "perspective-data"

    def __init__(self, **kwargs):
        super(PerspectiveMode, self).__init__(**kwargs)
        self._overlays = {}  # keyed by tdw
        self._last_used_point_uid = None
        self._last_added_point = None
        self._idle_srcid = None
        self._tdw = None
        self._model = None
        self.last_postion = None
        self._start_point = None
        self._sync_event_added = False
        self._overlay_widgets = OverlayWidgets()

    @property
    def options_presenter(self):
        """MVP presenter object for the perspective points editor panel"""
        cls = self.__class__
        if cls._OPTIONS_PRESENTER is None:
            cls._OPTIONS_PRESENTER = OptionsPresenter()
        if cls._OPTIONS_PRESENTER.target != self:
            cls._OPTIONS_PRESENTER.target = self
        return cls._OPTIONS_PRESENTER

    @property
    def inactive_cursor(self):
        "force default painting cursor"
        return None

    @property
    def active_cursor(self):
        "force default painting cursor"
        return None

    def enter(self, doc, **kwds):
        super(PerspectiveMode, self).enter(doc, **kwds)
        if not self._is_active():
            self._discard_overlays()
        tdw = self.doc.tdw
        self._ensure_overlay_for_tdw(tdw)
        if self._graph.is_empty():
            sdict = doc.model.settings.get(self._SETTINGS_KEY)
            self.get_options_widget()
            if sdict:
                self._graph.clear()
                self._graph.initialize_from_structure(sdict)
                self.options_presenter.create_from_graph(self._graph)
            else:
                self.reset_perspective('RectilinearProjection')
        self._update_visualizations()
        self.redraw_points_graph(self._graph.get_root())
        self._model = doc.model
        if not self._sync_event_added:
            doc.model.sync_pending_changes += self._sync_pending_changes_cb
        # doc.model.modified += self._doc_settings_modified_cb

    def _sync_pending_changes_cb(self, settings, flush=False):
        if not flush:
            return
        settings.settings[self._SETTINGS_KEY] = self._graph.to_object()

    def _is_active(self):
        for mode in self.doc.modes:
            if mode is self:
                return True
        return False

    def _discard_overlays(self):
        for tdw, overlay in self._overlays.items():
            tdw.display_overlays.remove(overlay)
            tdw.queue_draw()
        self._overlays.clear()

    def _ensure_overlay_for_tdw(self, tdw):
        overlay = self._overlays.get(tdw)
        if not overlay:
            overlay = Overlay(self, tdw)
            tdw.display_overlays.append(overlay)
            self._overlays[tdw] = overlay
        return overlay

    def leave(self, **kwds):
        if not self._is_active():
            self._discard_overlays()
        self._sync_pending_changes_cb(self._model, True)
        super(PerspectiveMode, self).leave(**kwds)

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        return self.options_presenter.widget

    def button_press_cb(self, tdw, event):
        model_x, model_y = tdw.display_to_model(event.x, event.y)
        is_pressed = self._overlay_widgets.button_press((model_x, model_y))
        if is_pressed:
            is_drag = self._overlay_widgets.is_drag()
            if is_drag:
                self._state.set_state(State.DRAG_POINT)
        return super(PerspectiveMode, self).button_press_cb(tdw, event)

    def button_release_cb(self, tdw, event):
        if self._state.is_add():
            model_x, model_y = tdw.display_to_model(event.x, event.y)
            self.add_point((model_x, model_y), 'VP')
        return super(PerspectiveMode, self).button_release_cb(tdw, event)

    def drag_start_cb(self, tdw, event):
        self._tdw = tdw
        self._model = tdw.doc
        self.update_position(event.x, event.y)
        if self._state.is_noop():
            self._state.set_state(State.DRAW)
            self.brushwork_rollback(tdw.doc)
            self.brushwork_begin(tdw.doc, abrupt=True)
            self._start_point = tdw.display_to_model(event.x, event.y)

    def drag_update_cb(self, tdw, event, x, y, dx, dy):
        self.update_position(x, y)
        if self._state.is_draw():
            if self._idle_srcid is None:
                self._idle_srcid = GLib.idle_add(self.process_line)
        elif self._state.is_drag():
            model_x, model_y = tdw.display_to_model(x, y)
            self._overlay_widgets.drag(
                self._graph, self._OPTIONS_PRESENTER, (model_x, model_y)
            )
            # TODO self._graph.update(point, new_position)
            tdw.renderer.defer_hq_rendering()

    # TODO rename process_line
    def process_line(self, forUpdte=True):
        "draw shape dependign on chosen tool"
        if self._state.is_draw():
            if self._state.is_line_tool():
                self._draw_line(forUpdte)
            elif self._state.is_circle_tool():
                self._draw_circle()
            elif self._state.is_polygon_tool():
                self._draw_polygon()
        self._idle_srcid = None

    # TODO rename update_position
    # TODO rewrite it - last_postion is bad
    def update_position(self, pos_x, pos_y):
        "set last used position"
        pos = self._tdw.display_to_model(pos_x, pos_y)
        self.last_postion = complex(pos[0], pos[1])

    def _update_visualizations(self):
        self._overlay_widgets.clear()
        for vis in self._graph.get_visualizations():
            if vis.type == 'point':
                point = self._graph.get_by_uid(vis.nodes[0])
                if not point.enabled:
                    continue
                self._overlay_widgets.add(OverlayPoint(point))
            elif vis.type == 'space':
                space = self._graph.get_by_uid(vis.nodes[0])
                config = self._graph.get_by_uid(vis.nodes[1])
                if not space.enabled:
                    continue
                self._overlay_widgets.add(OverlaySpace(space, config))
            elif vis.type == 'projection':
                view = self._graph.get_by_uid(vis.nodes[0])
                config = self._graph.get_by_uid(vis.nodes[1])
                if not view.enabled:
                    continue
                self._overlay_widgets.add(OvarlayProjection(view, config))
            else:
                logger.info('Unsupported visualization type %r', vis['type'])

    def redraw_points_graph(self, node=None):
        "redraw points, children of node"
        node = node or self._graph.get_root()
        for point in self._graph.get_points(node):
            point_pos = point.get_position()
            self._queue_draw_node(point_pos)

    def _queue_draw_node(self, pos):
        """Redraws a specific control node on all known view TDWs"""
        if pos is None:
            return
        for tdw in self._overlays:
            if math.fabs(pos.real) > 1000000 or math.fabs(pos.imag) > 1000000:
                continue
            display_x, display_y = tdw.model_to_display(pos.real, pos.imag)
            display_x = math.floor(display_x)
            display_y = math.floor(display_y)
            size = math.ceil(gui.style.DRAGGABLE_POINT_HANDLE_SIZE * 2)
            tdw.queue_draw_area(
                display_x - size, display_y - size,
                size * 2 + 1, size * 2 + 1
            )

    def drag_stop_cb(self, tdw):
        if self._state.is_draw():
            self.process_line(forUpdte=False)
            self.brushwork_commit(tdw.doc)
        if self._state.is_drag():
            self._overlay_widgets.drag_stop()
        self._idle_srcid = None
        self._state.set_state(State.NOOP)

    def add_point(self, position, name=None):
        "add point"
        point_data = get_default_data('point')
        update_data(point_data, '1', {
            'name': name,
            'position': position
        })
        point = self._graph.add_sub_graph(point_data)

        self._last_added_point = point
        self._update_visualizations()
        self._queue_draw_node(complex(position[0], position[1]))
        self.options_presenter.create_from_graph(self._graph, [point])
        return point

    def remove_point(self, point_unique_id):
        "remove point"
        point = self._graph.remove_by_uid(point_unique_id)
        self.redraw_points_graph(point)
        self.set_chosen_point(None)

    def toggle_point(self, node_uid):
        "toggle visibility of graph subtree"
        point = self._graph.get_by_uid(node_uid)
        point.toggle()

        def toggle_fct(point):
            "toggle visibility of point"
            point.update_toggle_from_parent()
            self.redraw_points_graph(point)

        for node in self._graph.get_all_nodes(point):
            toggle_fct(node)

    def lock_point(self, node_uid):
        "toggle edit possibility for graph subtree"
        point = self._graph.get_by_uid(node_uid)
        point.lock()

        for node in self._graph.get_all_nodes(point):
            node.update_lock_from_parent()

    # TODO rename get_point_by_uid
    def get_point_by_uid(self, point_unique_id):
        "get node by uid"
        if point_unique_id is None:
            return None
        return self._graph.get_by_uid(point_unique_id)

    # TODO only for its name is it used
    def get_last_used_point(self):
        "get last used point"
        return self.get_point_by_uid(self._last_used_point_uid)

    def set_chosen_point(self, point_unique_id):
        "set chosen point"
        if point_unique_id is None:
            self._graph.chosen_point = None
            return

        chosen_point = self._graph.get_by_uid(point_unique_id)
        self._graph.chosen_point = chosen_point
        self.redraw_points_graph()

    def on_selection_changed(self, selected_uids):
        "update conditions for context menu and set chosen point"
        vanishig_points = p.NodeWrapperVector()
        selected = []
        for uid in selected_uids:
            node = self._graph.get_by_uid(uid)
            selected.append(node)
            if node.is_vanishing_point():
                vanishig_points.append(node)

        plane = self._graph.get_by_tag('__implicit_plane')
        selected_count = len(selected)

        conditions = set()

        if selected_count == 1:
            single_element = selected[0]
            self.set_chosen_point(single_element.uid)
            conditions.add('single')
            if single_element.get_view().is_projection():
                conditions.add('projection')
        elif selected_count == 2:
            conditions.add('two')
        else:
            self.set_chosen_point(None)

        if selected_count == 0:
            conditions.add('none')

        if selected_count <= 2:
            plane.update_compute_point_source(self._graph, vanishig_points)
        else:
            plane.update_compute_point_source(self._graph, [])

        self.options_presenter.actions_set_enabled(conditions)

    def _get_best_fitting_line(self):
        origin = self._start_point
        directions = []

        for point in self._graph.get_all_enabled_points():
            line = point.get_line(complex(origin[0], origin[1]))
            directions.append((point.uid, line))

        best_dir = [float("inf"), None, None]

        if not directions:
            return None

        for uid, line in directions:
            line_distance = line.get_distance(self.last_postion)
            if best_dir[0] > line_distance:
                best_dir = [line_distance, uid, line]

        self._last_used_point_uid = best_dir[1]

        return best_dir[2].get_line_points(self.last_postion)

    def _draw_line(self, forUpdte=False):
        model = self._model

        line_data = self._get_best_fitting_line()

        if not line_data:
            return

        line_box = get_line_bounding_box(line_data)
        line_width = 2
        for tdw in self._overlays:
            min_x, min_y = tdw.model_to_display(line_box.min_x, line_box.min_y)
            max_x, max_y = tdw.model_to_display(line_box.max_x, line_box.max_y)
            # TODO FIXME to nie działa, linia jest przycięta
            tdw.queue_draw_area(
                math.floor(min_x) - line_width,
                math.floor(min_y) - line_width,
                math.floor(max_x - min_x) + line_width + line_width,
                math.floor(max_y - min_y) + line_width + line_width,
            )

        if not forUpdte:
            self.brushwork_rollback(model)
            self.brushwork_begin(model, abrupt=True)

            self._draw_lines(model, line_data, False)

    def _draw_polygon_imp(self, side_count, side_teselation=1):
        default_projection = self._graph.main_view
        model = self._model
        self.brushwork_rollback(model)
        self.brushwork_begin(model, abrupt=True)
        center = self._start_point
        start = self.last_postion
        plane = self._graph.get_by_tag('__implicit_plane').as_plane()
        projection = default_projection.as_projection()
        start_ray = projection.calc_direction(start)
        center_ray = projection.calc_direction(complex(center[0], center[1]))

        plane_normal = plane.get_normal()
        start_ray_dot_sign = plane_normal.dot_3D(start_ray)
        center_ray_dot_sign = plane_normal.dot_3D(center_ray)
        # test if start point is on other side of horizon than center point
        if start_ray_dot_sign * center_ray_dot_sign <= 0:
            return

        center_3d_pos = projection.intersect_view_ray_canvas(center_ray)
        start_3d_pos = intersect_view_ray_and_plane(
            plane_normal, center_3d_pos, start_ray
        )

        circle = p.create_circle(
            center_3d_pos, start_3d_pos, plane_normal, side_count
        )

        if side_teselation > 1:
            circle = teselate_poligon(circle, side_teselation)

        canvas_circle = projection.project_on_canvas(circle)

        self._draw_lines(model, canvas_circle, True)

    def _draw_circle(self):
        self._draw_polygon_imp(64)

    def _draw_polygon(self):
        self._draw_polygon_imp(self._state.polygon_side_count, 100)

    def set_polygon_side_count(self, side_count):
        "set polygon side count for polygon tool"
        self._state.polygon_side_count = int(side_count)

    def _trim_points(self, points):
        tdw = self.doc.tdw
        alloc = tdw.get_allocation()
        corner = (tdw.display_to_model(0, 0))
        corner2 = (tdw.display_to_model(alloc.width, alloc.height))

        (width, height) = (
            corner2[0] - corner[0],
            corner2[1] - corner[1]
        )

        min_x = min(-2000, corner[0] - height)
        max_x = max(2000, corner[0] + 2 * height)
        min_y = min(-2000, corner[1] - width)
        max_y = max(2000, corner[1] + 2 * width)

        trimmed = []

        for pos in points:
            pos_x = max(min(pos.real, max_x), min_x)
            pos_y = max(min(pos.imag, max_y), min_y)
            trimmed.append((pos_x, pos_y))

        return trimmed

    def _draw_lines(self, model, points, is_loop=False):
        points = self._trim_points(points)

        viewzoom = self.doc.tdw.scale
        viewrotation = self.doc.tdw.rotation

        if is_loop:
            self.stroke_to(
                model, 0.1, points[-1][0], points[-1][1], 0.5, 0.0, 0.0,
                viewzoom, viewrotation, 0.0,
                auto_split=False,
            )
        for point in points:
            self.stroke_to(
                model, 0.1, point[0], point[1], 0.5, 0.0, 0.0,
                viewzoom, viewrotation, 0.0,
                auto_split=False,
            )

    def reset_perspective(self, projection_type):
        "reset perspective tool to chosen preset"
        # TODO cleaning old root
        self.options_presenter.clear()
        self._graph.clear()
        self._overlay_widgets.clear()

        view_data = self.create_perspective(projection_type)

        self._graph.initialize_from_structure(view_data)
        self.options_presenter.create_from_graph(self._graph)
        self._update_visualizations()
        self.redraw_points_graph(self._graph.get_root())

    def add_perspective(self, projection_type):
        """
        add new perspective root node to treeview.
        perspective node support multiple active perspectives
        """
        view_data = self.create_perspective(projection_type)

        view_node = self._graph.add_sub_graph(view_data)
        self.options_presenter.create_from_graph(self._graph, [view_node])
        self._update_visualizations()
        self.redraw_points_graph(view_node)

    def create_perspective(self, projection_type):
        "create perspective"
        tdw = self.doc.tdw
        alloc = tdw.get_allocation()
        pos_y = alloc.y + alloc.height / 2
        x_offset = alloc.width * 0.1
        left = (tdw.display_to_model(alloc.x + x_offset, pos_y))
        right = (tdw.display_to_model(alloc.x + alloc.width - x_offset, pos_y))
        return self._create_view_data_from_points(left, right, projection_type)

    def reset_preset(self, preset_name='perspective_projection'):
        "reset preset"
        if preset_name == 'perspective_projection':
            self.reset_perspective('RectilinearProjection')
        elif preset_name == 'directions':
            self.reset_perspective('RectilinearProjection')
        elif preset_name == 'curvilinear_perspective_projection':
            self.reset_perspective('CurvilinearPerspective')

    def add_preset(self, preset_name='perspective_projection'):
        "reset preset"
        if preset_name == 'perspective_projection':
            self.add_perspective('RectilinearProjection')
        elif preset_name == 'directions':
            self.add_perspective('RectilinearProjection')
        elif preset_name == 'curvilinear_perspective_projection':
            self.add_perspective('CurvilinearPerspective')

    # TODO make function form this _create_view_data_from_points
    def _create_view_data_from_points(self, pos1, pos2, projection_type):
        center = ((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2)
        data = get_default_data('perspective_projection')
        update_data(data, 'View', {
            'left': pos1,
            'right': pos2,
            'type': projection_type
        })
        update_data(data, 'SP_View', {
            'center': center
        })
        update_data(data, 'SP_V_Conf', {
            'center': center,
        })
        update_data(data, 'SP_C', {
            'position': center
        })
        update_data(data, 'SP_S', {
            'position': pos2
        })
        update_data(data, 'SP_R', {
            'position': pos1
        })
        return data

    def create_group(self, name=None):
        "create group of nodes"
        group_data = get_default_data('group')
        if name is not None:
            update_data(group_data, 'group', {
                'name': name,
            })
        group_node = self._graph.add_sub_graph(group_data)
        self.options_presenter.create_from_graph(self._graph, [group_node])
        self.redraw_points_graph(group_node)
        self._update_visualizations()
        return group_node

    def create_space(self, axis=None, angle=None, name=None):
        "create space, space can transform its children nodes"
        axis = axis or (0, 1, 0)
        angle = angle or 0
        name = name or _('2P')
        space_data = get_default_data('space')
        update_data(space_data, 'space', {
            'name': name,
            'up': axis,
            'rotation_angle': angle
        })
        update_data(space_data, 'Forward', {
            'role': 'SPACE'
        })
        update_data(space_data, 'Side', {
            'role': 'SPACE'
        })
        p2_space = self._graph.add_sub_graph(space_data)

        self.options_presenter.create_from_graph(self._graph, [p2_space])
        self._update_visualizations()
        self.redraw_points_graph(p2_space)
        return p2_space

    # TODO compute - nie lubię tego słowa
    # TODO docstring - rozwinąć
    def create_compute_point(self, parents_uids, fct_name, point_params):
        "create compute point"
        src_nodes = p.NodeWrapperVector()
        for parent_uid in parents_uids:
            src_node = self._graph.get_by_uid(parent_uid)
            if not src_node.is_point():
                return
            src_nodes.append(src_node)

        for params in point_params:
            separator = params.get('separator', ' ')
            dst_name = separator.join([i.name for i in src_nodes])
            dst_name = dst_name + ' ' + params.get('suffix', '')
            dst_node = self.add_point((0, 0), dst_name)
            additional = 0  # TODO allow more parameters
            if 'additional' in params:
                additional = params['additional'][0]
            self._graph.convert_to_compute_node(
                dst_node, src_nodes, fct_name, additional
            )
            point_pos = dst_node.get_position()
            self._queue_draw_node(point_pos)

    def get_overlay_widgets(self):
        for vis in self._overlay_widgets.getAll():
            yield vis
        if self._state.is_draw():
            line_data = self._get_best_fitting_line()
            if not line_data:
                return
            if not len(line_data) >= 2:
                return
            yield OverlayLine(line_data)


class OverlayBase(object):
    def __init__(self):
        pass

    def test_button_press(self, model_pos):
        raise NotImplementedError

    # TODO better graph handling
    def drag(self, graph, model_pos):
        raise NotImplementedError

    def paint(self, tdw, cr):
        raise NotImplementedError

    def dependences(self):
        raise NotImplementedError

    def drag_nodes_to_update(self):
        return ()


class OverlayLine(OverlayBase):
    def __init__(self, segments):
        self._segments = segments

    def test_button_press(self, model_pos):
        pass

    def paint(self, tdw, cr):
        begin_x, begin_y = tdw.model_to_display(
            self._segments[0].real, self._segments[0].imag
        )
        cr.save()
        cr.move_to(begin_x, begin_y)
        for pos in self._segments[1:]:
            pos_x, pos_y = tdw.model_to_display(pos.real, pos.imag)
            cr.line_to(pos_x, pos_y)
        cr.stroke()
        cr.restore()


class OverlayPoint(OverlayBase):
    def __init__(self, point):
        super(OverlayPoint, self).__init__()
        self._point = point

    def test_button_press(self, model_pos):
        if not (self._point.parent_enabled and self._point.enabled):
            return False
        if self._point.parent_locked or self._point.locked:
            return False
        point_pos = self._point.get_position()
        delta = (
            model_pos[0] - point_pos.real,
            model_pos[1] - point_pos.imag
        )
        squared_distance = delta[0] * delta[0] + delta[1] * delta[1]
        if squared_distance < 100:
            return True
        return False

    def drag(self, graph, model_pos):
        graph.update(self._point, complex(model_pos[0], model_pos[1]))

    def drag_nodes_to_update(self):
        return [self._point]

    def paint(self, tdw, cr):
        if not (self._point.parent_enabled and self._point.enabled):
            return
        model_pos = self._point.get_position()
        pos_x, pos_y = tdw.model_to_display(
            model_pos.real, model_pos.imag
        )
        color = gui.style.EDITABLE_ITEM_COLOR
        if self._point.color:
            color = self._point.color
            rgb = (
                (color >> 24) / 255.0,          # R
                ((color >> 16) & 255) / 255.0,  # G
                ((color >> 8) & 255) / 255.0,   # B
            )
            color = RGBColor(rgb=rgb)
        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        gui.drawutils.render_round_floating_color_chip(
            cr=cr, x=pos_x, y=pos_y,
            color=color,
            radius=radius,
        )


class OvarlayProjection(OverlayBase):
    def __init__(self, view, config):
        super(OvarlayProjection, self).__init__()
        self._view = view
        self._config = config
        self._drag = None

    def test_button_press(self, model_pos):
        if not (self._config.parent_enabled and self._config.enabled):
            return False
        if self._config.parent_locked or self._config.locked:
            return False

        model_pos = complex(model_pos[0], model_pos[1])

        view = self._view.as_projection()
        center = view.get_center_complex()
        if abs(center-model_pos) < 10:
            self._drag = "center"
            return True
        size = view.get_size()
        rotation = view.get_rotation()
        if abs(center + complex(1, 0)*rotation*size-model_pos) < 10:
            self._drag = "size"
            return True
        if abs(center + complex(-1, 0)*rotation*size-model_pos) < 10:
            self._drag = "rotation"
            return True
        return False

    def drag(self, graph, model_pos):
        view = self._view.as_projection()
        model_pos = complex(model_pos[0], model_pos[1])
        if self._drag == "center":
            view.set_center(model_pos)
        elif self._drag == "size":
            center = view.get_center_complex()
            view.set_size(abs(center - model_pos))
        elif self._drag == "rotation":
            center = view.get_center_complex()
            rotation = center - model_pos
            # save normalized rotation
            view.set_rotation(rotation/abs(rotation))
        graph.update_groups(self._view)

    def paint(self, tdw, cr):
        if not (self._config.parent_enabled and self._config.enabled):
            return
        view = self._view.as_projection()
        points = []
        center = view.get_center_complex()
        points.append(center)
        size = view.get_size()
        rotation = view.get_rotation()
        points.append(center + complex(1, 0)*rotation*size)
        points.append(center + complex(-1, 0)*rotation*size)

        for point in points:
            pos_x, pos_y = tdw.model_to_display(point.real, point.imag)
            color = gui.style.EDITABLE_ITEM_COLOR
            radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
            gui.drawutils.render_round_floating_color_chip(
                cr=cr, x=pos_x, y=pos_y,
                color=color,
                radius=radius,
            )


class OverlaySpace(OverlayBase):
    def __init__(self, space, config):
        super(OverlaySpace, self).__init__()
        self._space = space
        self._config = config
        # in normalized perspective projection plane
        self._cube_corner = complex(0, 0.1)
        # in normalized perspective space
        self._cube_size = 0.5
        self._cube_size_up = 0.1
        self._cube_size_forward = 0.1
        self._cube_size_side = 0.1

    def test_button_press(self, model_pos):
        if not (self._config.parent_enabled and self._config.enabled):
            return False
        if self._config.parent_locked or self._config.locked:
            return False
        space = self._space.as_space()
        view = self._space.get_view().as_projection()
        center = view.get_center_complex()

        corner = (
            center + self._cube_corner * view.get_rotation() * view.get_size()
        )

        corner_ray = view.calc_direction(corner)
        corner_3d = view.intersect_view_ray_canvas(corner_ray)

        points_cube = []
        points_cube.append(Quaternion(0, 0, 0, 0))   # corner
        points_cube.append(Quaternion(0, 0, 1, 0))   # froward
        points_cube.append(Quaternion(0, 1, 0, 0))   # up
        points_cube.append(Quaternion(1, 0, 0, 0))   # side

        drag = ("corner", "forward", "up", "side")

        for index, point in enumerate(points_cube):
            point_3d = corner_3d + p.rotate(
                space.get_rotation(), point.scalar_mul(self._cube_size)
            )
            position = view.calc_pos_from_dir(point_3d)
            diff = (position.real - model_pos[0], position.imag - model_pos[1])
            if diff[0] * diff[0] + diff[1] * diff[1] < 100:
                self._drag = drag[index]
                return True

        return False

    def drag(self, graph, model_pos):
        if self._drag == "corner":
            view = self._space.get_view().as_projection()
            internal_pos = view.model_position_to_internal(
                complex(model_pos[0], model_pos[1])
            )
            self._cube_corner = internal_pos
        elif self._drag == "up":
            space = self._space.as_space()
            view = self._space.get_view().as_projection()
            center = view.get_center_complex()
            up = p.rotate(space.get_rotation(), Quaternion(0, 1, 0, 0))

            corner = (
                center
                + self._cube_corner * view.get_rotation() * view.get_size()
            )
            corner_ray = view.calc_direction(corner)
            corner_3d = view.intersect_view_ray_canvas(corner_ray)

            drag_ray = view.calc_direction(complex(model_pos[0], model_pos[1]))
            drag_3d = view.intersect_view_ray_canvas(drag_ray)

            offset_3d = corner_3d - drag_3d
            length = up.dot_3D(offset_3d)
            self._cube_size = -length   # FIXME why `-`?

    def _paint_horizons(self, tdw, cr):
        space = self._space.as_space()
        view = self._space.get_view().as_projection()

        forward = Quaternion(0, 0, 1, 0)
        up = Quaternion(0, 1, 0, 0)
        side = Quaternion(1, 0, 0, 0)

        for base_direction in (forward, up, side):
            direction = p.rotate(space.get_rotation(), base_direction)

            horizon = view.get_horizon_line(direction)
            alloc = tdw.get_allocation()
            corner = (tdw.display_to_model(0, 0))
            corner2 = (tdw.display_to_model(alloc.width, alloc.height))
            horizon_points = horizon.for_bbox(
                complex(corner[0], corner[1]), complex(corner2[0], corner2[1])
            )

            if horizon_points:
                begin_x, begin_y = tdw.model_to_display(
                    horizon_points[0].real, horizon_points[0].imag
                )
                cr.save()
                cr.move_to(begin_x, begin_y)
                for pos in horizon_points[1:]:
                    pos_x, pos_y = tdw.model_to_display(pos.real, pos.imag)
                    cr.line_to(pos_x, pos_y)
                cr.stroke()
                cr.restore()

    def _paint_cube(self, tdw, cr):
        space = self._space.as_space()
        view = self._space.get_view().as_projection()
        center = view.get_center_complex()

        corner = (
            center + self._cube_corner * view.get_rotation() * view.get_size()
        )

        corner_ray = view.calc_direction(corner)
        corner_3d = view.intersect_view_ray_canvas(corner_ray)

        points_cube = []
        points_cube.append(Quaternion(0, 0, 0, 0))   # corner
        points_cube.append(Quaternion(0, 0, 1, 0))   # froward
        points_cube.append(Quaternion(0, 1, 0, 0))   # up
        points_cube.append(Quaternion(1, 0, 0, 0))   # side
        points_cube.append(Quaternion(1, 1, 0, 0))
        points_cube.append(Quaternion(1, 0, 1, 0))
        points_cube.append(Quaternion(0, 1, 1, 0))
        points_cube.append(Quaternion(1, 1, 1, 0))

        positions_2d = []
        points_3d = []
        for point in points_cube:
            point_3d = corner_3d + p.rotate(
                space.get_rotation(), point.scalar_mul(self._cube_size)
            )
            points_3d.append(point_3d)
            position = view.calc_pos_from_dir(point_3d)
            positions_2d.append(position)

        poligons = []

        tmp_vector = p.QuaternionVector()
        tmp_vector.append(points_3d[0])
        tmp_vector.append(points_3d[2])
        tmp_vector.append(points_3d[4])
        tmp_vector.append(points_3d[3])
        poligons.append(tmp_vector)

        tmp_vector = p.QuaternionVector()
        tmp_vector.append(points_3d[1])
        tmp_vector.append(points_3d[5])
        tmp_vector.append(points_3d[7])
        tmp_vector.append(points_3d[6])
        poligons.append(tmp_vector)

        tmp_vector = p.QuaternionVector()
        tmp_vector.append(points_3d[0])
        tmp_vector.append(points_3d[1])
        poligons.append(tmp_vector)

        tmp_vector = p.QuaternionVector()
        tmp_vector.append(points_3d[2])
        tmp_vector.append(points_3d[6])
        poligons.append(tmp_vector)

        tmp_vector = p.QuaternionVector()
        tmp_vector.append(points_3d[3])
        tmp_vector.append(points_3d[5])
        poligons.append(tmp_vector)

        tmp_vector = p.QuaternionVector()
        tmp_vector.append(points_3d[4])
        tmp_vector.append(points_3d[7])
        poligons.append(tmp_vector)

        for poligon in poligons:
            teselated = teselate_poligon(poligon, 10)
            canvas_poligon = view.project_on_canvas(teselated)
            if not len(canvas_poligon):
                continue

            begin_x, begin_y = tdw.model_to_display(
                canvas_poligon[-1].real, canvas_poligon[-1].imag
            )
            cr.save()
            cr.move_to(begin_x, begin_y)
            for pos in canvas_poligon:
                pos_x, pos_y = tdw.model_to_display(pos.real, pos.imag)
                cr.line_to(pos_x, pos_y)
            cr.stroke()
            cr.restore()

        for position in positions_2d[:4]:
            pos_x, pos_y = tdw.model_to_display(position.real, position.imag)
            color = gui.style.EDITABLE_ITEM_COLOR
            radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
            gui.drawutils.render_round_floating_color_chip(
                cr=cr, x=pos_x, y=pos_y,
                color=color,
                radius=radius,
            )

    def paint(self, tdw, cr):
        if not (self._config.parent_enabled and self._config.enabled):
            return False
        self._paint_horizons(tdw, cr)
        self._paint_cube(tdw, cr)


class Overlay(gui.overlays.Overlay):
    def __init__(self, perspectiveMode, tdw):
        super(Overlay, self).__init__()
        self._perspective = weakref.proxy(perspectiveMode)
        self._tdw = weakref.proxy(tdw)

    def paint(self, cr):
        for widget in self._perspective.get_overlay_widgets():
            widget.paint(self._tdw, cr)

        point = self._perspective.get_last_used_point()

        # This part should be extracted to separate function,
        # most of its code come from  gui.overlays.ScaleOverlay
        if self._perspective._state.is_draw() and point:
            text = point.name
            layout = self._tdw.create_pango_layout(text)

            # Set a bold font
            font = layout.get_font_description()
            if font is None:  # inherited from context
                font = layout.get_context().get_font_description()
                font = font.copy()
            layout.set_font_description(font)

            # General dimensions
            alloc = self._tdw.get_allocation()
            lw, lh = layout.get_pixel_size()

            # Background rectangle
            m = 10
            h = alloc.height
            p = 10
            area = bx, by, bw, bh = (
                m,
                h - p - m - lh - p,
                lw + p + p,
                lh + p + p
            )
            rounded_box(cr, bx, by, bw, bh, p)
            rgba = list(gui.style.TRANSIENT_INFO_BG_RGBA)
            rgba[3] *= 1
            cr.set_source_rgba(*rgba)
            cr.fill()

            # Text
            cr.translate(bx + p, by + p)
            rgba = list(gui.style.TRANSIENT_INFO_RGBA)
            cr.set_source_rgba(*rgba)
            PangoCairo.show_layout(cr, layout)

            self._tdw.queue_draw_area(*area)


# TODO wydzielić do oddzielnego pliku
class OptionsPresenter(object):
    """Presents UI for editing perspective configuration"""
    def __init__(self):
        super(OptionsPresenter, self).__init__()
        self._options_grid = None
        self.target = None
        self.actions_conditions = dict()
        self.present_conditions = dict()

    class STORE_COLUMNS:
        ID = 0
        NAME = 1
        NOOP = 2  # force update workaround
        ANGLE = 3
        COLOR = 4

    @property
    def widget(self):
        if self._options_grid is None:
            builder_xml = os.path.splitext(__file__)[0] + ".glade"
            builder_menu_xml = os.path.splitext(__file__)[0] + "_menu.xml"
            builder = Gtk.Builder()
            builder.set_translation_domain("mypaint")
            builder.add_from_file(builder_xml)
            builder.add_from_file(builder_menu_xml)
            builder.connect_signals(self)

            self._builder = builder
            self._options_grid = builder.get_object("perspective_option_grid")
            self._points_store = builder.get_object("points_store")
            self._points_treeview = builder.get_object("points_treeview")
            self._selection = self._points_treeview.get_selection()

            locked_column = builder.get_object("locked_column")
            locked_cell = builder.get_object("locked_cell")
            locked_column.set_cell_data_func(
                locked_cell, self.locked_pixbuf_datafunc
            )
            self._locked_column = locked_column

            active_column = builder.get_object("active_column")
            active_cell = builder.get_object("active_cell")
            active_column.set_cell_data_func(
                active_cell, self.active_pixbuf_datafunc
            )
            self._active_column = active_column

            color_column = builder.get_object("color_column")
            color_cell = builder.get_object("color_cell")
            color_column.set_cell_data_func(
                color_cell, self.color_pixbuf_datafunc
            )
            self._color_column = color_column

            self._add_3p_window = builder.get_object("add_3p_window")
            add_3p_edit_cancel_button = builder.get_object(
                "3P_edit_cancel_button"
            )
            add_3p_edit_cancel_button.connect(
                'clicked', self._hide_window_on_click_cb, self._add_3p_window
            )

            gmenu = builder.get_object("point_list_menu")
            self.point_list_menu = Gtk.Menu.new_from_model(gmenu)

            self._preset_combo_box = builder.get_object("preset_combo_box")
            self._preset_combo_box.set_active(0)
            self._preset_store = builder.get_object("presets_store")

            self.init_actions()

        return self._options_grid

    def init_actions(self):
        action_group = Gio.SimpleActionGroup.new()

        actions = [(
            'add', self._add_point_action_cb,
            lambda cond: not cond.isdisjoint(set(['projection', 'none']))
        ), (
            'delete', self._delete_point_action_cb,
            lambda cond: 'single' in cond
        ), (
            'add_group', self._add_group_action_cb,
            lambda cond: not cond.isdisjoint(set(['projection', 'none']))
        ), (
            'add_2P', self._2P_action_cb,
            lambda cond: not cond.isdisjoint(set(['projection', 'none']))
        ), (
            'add_3P', self._3P_action_cb,
            lambda cond: not cond.isdisjoint(set(['projection', 'none']))
        ), (
            'add_MP', self._create_MP_action_cb,
            lambda cond: cond >= set(['single', 'projection'])
        ), (
            'add_MP_2', self._create_MP_2_action_cb,
            lambda cond: 'two' in cond
        ), (
            'add_mirrored', self._create_mirrored_action_cb,
            lambda cond: cond >= set(['single', 'projection'])
        ), (
            'add_cross_product', self._create_cross_product_action_cb,
            lambda cond: 'two' in cond
        ), (
            'add_2d_direction', self._create_2d_direction_action_cb,
            lambda cond: 'two' in cond
        ), (
            'add_2d_direction_90', self._create_2d_direction_90_action_cb,
            lambda cond: 'two' in cond
        )]

        for (name, callback, conditions) in actions:
            action = Gio.SimpleAction.new_stateful(
                name, None, GLib.Variant('u', 0)
            )
            action.connect('activate', callback)
            action.connect('change-state', self._action_set_state_cb)
            action_group.add_action(action)
            self.actions_conditions[name] = conditions

        self.point_list_menu.insert_action_group('listMenu', action_group)

        point_list_toolbar = self._builder.get_object("point_list_toolbar")
        styles = point_list_toolbar.get_style_context()
        styles.add_class(Gtk.STYLE_CLASS_INLINE_TOOLBAR)
        point_list_toolbar.insert_action_group("toolbar", action_group)
        self.action_group = action_group

    def actions_set_enabled(self, conditions):
        self.present_conditions = conditions
        for action_name in self.action_group.list_actions():
            self.action_group.change_action_state(
                action_name, GLib.Variant('u', 1)
            )

    def _action_set_state_cb(self, action, value):
        action_name = action.get_name()
        action.set_enabled(
            self.actions_conditions[action_name](self.present_conditions)
        )

    def locked_pixbuf_datafunc(self, column, cell, model, it, data):
        """Use a padlock icon to show point is locked"""
        del column, model, data
        uid = self._points_store.get_value(it, self.STORE_COLUMNS.ID)
        point = self.target.get_point_by_uid(uid)
        if point.locked:
            icon_name = "mypaint-object-locked-symbolic"
        else:
            icon_name = "mypaint-object-unlocked-symbolic"
        cell.set_property("icon-name", icon_name)
        cell.set_property("sensitive", not point.parent_locked)

    def active_pixbuf_datafunc(self, column, cell, model, it, data):
        """Use a eye icon to show point is active"""
        del column, model, data
        uid = self._points_store.get_value(it, self.STORE_COLUMNS.ID)
        point = self.target.get_point_by_uid(uid)
        if point.enabled:
            icon_name = "mypaint-object-visible-symbolic"
        else:
            icon_name = "mypaint-object-hidden-symbolic"
        cell.set_property("icon-name", icon_name)
        cell.set_property("sensitive", point.parent_enabled)

    def color_pixbuf_datafunc(self, column, cell, model, it, data):
        """Show color chosen for element"""
        del column, model, data
        uid = self._points_store.get_value(it, self.STORE_COLUMNS.ID)
        point = self.target.get_point_by_uid(uid)
        dst = None
        if not point.is_point():
            dst = GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB, 1, 8, 1, 1)
        else:
            dst = GdkPixbuf.Pixbuf.new(
                GdkPixbuf.Colorspace.RGB, False, 8, 16, 16
            )
        color = self._points_store.get_value(it, self.STORE_COLUMNS.COLOR)
        dst.fill(color)
        cell.set_property("pixbuf", dst)

    def _add_point_action_cb(self, action, value):
        self.target._state.set_state(State.ADD_POINT)

    def get_selected_point(self):
        selection = self._selection
        point_model, paths = selection.get_selected_rows()

        for path in paths:
            point_iter = point_model.get_iter(path)
            point_unique_id = point_model.get_value(
                point_iter,
                self.STORE_COLUMNS.ID
            )
            return (point_unique_id, point_iter)
        return (None, None)

    def get_selected_points(self):
        selection = self._selection
        point_model, paths = selection.get_selected_rows()
        points = []

        for path in paths:
            point_iter = point_model.get_iter(path)
            point_unique_id = point_model.get_value(
                point_iter,
                self.STORE_COLUMNS.ID
            )
            points.append((point_unique_id, point_iter))
        return points

    def get_point_by_uid(self, uid):
        point_model = self._points_treeview.get_model()
        point_iter = point_model.get_iter_first()
        go_up = False
        while point_iter:
            if go_up:
                go_up = False
                point_iter = point_model.iter_parent(point_iter)
                if point_iter:
                    point_iter_tmp = point_model.iter_next(point_iter)
                    if point_iter_tmp:
                        point_iter = point_iter_tmp
                    else:
                        go_up = True
                continue

            point_unique_id = point_model.get_value(
                point_iter,
                self.STORE_COLUMNS.ID
            )
            if point_unique_id == uid:
                return point_iter
            else:
                if point_model.iter_has_child(point_iter):
                    point_iter = point_model.iter_children(point_iter)
                else:
                    point_iter_tmp = point_model.iter_next(point_iter)
                    if point_iter_tmp:
                        point_iter = point_iter_tmp
                    else:
                        go_up = True
        return None

    def _delete_point_action_cb(self, action, value):
        (uid, point_iter) = self.get_selected_point()
        if uid is None:
            return
        self.target.remove_point(uid)
        self._points_store.remove(point_iter)

    def add_element_to_store(self, point, parent_uid=None):
        unique_id = point.uid
        name = point.name
        angle = point.get_description()
        parent = None
        if parent_uid:
            parent = self.get_point_by_uid(parent_uid)
        color = 0xFFFFFF00
        if point.color:
            color = point.color
        return self._points_store.append(
            parent,
            (unique_id, name, 0, angle, color)
        )

    def clear(self):
        self._points_store.clear()

    def remove_from_graph(self, nodes):
        for node in nodes:
            node_iter = self.get_point_by_uid(node.uid)
            if node_iter:
                self._points_store.remove(node_iter)

    def create_from_graph(self, graph, nodes=None):
        if not nodes:
            nodes = []
            for child in graph.get_root().get_children():
                nodes.append(child)

        while nodes:
            node = nodes.pop()
            for child in node.get_children():
                nodes.append(child)
            if not node.is_UI():
                continue
            ui_parent = graph.find_UI_parent(node)
            parent_uid = ui_parent and ui_parent.uid
            self.add_element_to_store(node, parent_uid)

    def update_from_graph(self, graph, nodes):
        self.remove_from_graph(nodes)
        self.create_from_graph(graph, nodes)

    def _points_selection_changed_cb(self, selection):
        model, selected = selection.get_selected_rows()
        point_unique_id = None
        selected_uids = set()
        for path in selected:
            point_iter = model.get_iter(path)
            point_unique_id = model.get_value(
                point_iter,
                self.STORE_COLUMNS.ID
            )
            selected_uids.add(point_unique_id)
        self.target.on_selection_changed(selected_uids)
        self.target.set_chosen_point(point_unique_id)

    def _point_edit_action(self):
        (uid, point_iter) = self.get_selected_point()
        if uid is None:
            return
        self._selected_point_uid = uid
        point = self.target.get_point_by_uid(uid)
        if point is None:
            return

        name = self._points_store.get_value(
            point_iter,
            self.STORE_COLUMNS.NAME
        )
        new_name = gui.dialogs.ask_for_name(
            self._options_grid,
            _("Name"),
            name
        )
        if new_name:
            point.name = new_name
            self._points_store.set_value(
                point_iter, self.STORE_COLUMNS.NAME, new_name
            )

    def _hide_window(self, window):
        window.set_modal(False)
        window.hide()
        return True

    def _hide_window_on_click_cb(self, button, window):
        return self._hide_window(window)

    def _hide_on_close_cb(self, widget, event):
        return self._hide_window(widget)

    def for_each_in_subtree(self, row_iter, fct):
        find_children = True
        first_row_iter = row_iter
        point_model = self._points_treeview.get_model()
        while row_iter:
            if find_children and point_model.iter_has_child(row_iter):
                row_iter = point_model.iter_children(row_iter)
            else:
                fct(row_iter)
                if row_iter == first_row_iter:
                    break
                row_iter_tmp = point_model.iter_next(row_iter)
                if row_iter_tmp:
                    find_children = True
                    row_iter = row_iter_tmp
                else:
                    find_children = False
                    row_iter = point_model.iter_parent(row_iter)

    def _touch_subtree(self, row_iter):
        self.for_each_in_subtree(
            row_iter,
            lambda x: self._points_store.set_value(
                x,
                self.STORE_COLUMNS.NOOP,
                0
            )
        )

    def update_description(self, uid):
        point = self.target.get_point_by_uid(uid)
        if not point.is_point():
            return
        point_model = self._points_treeview.get_model()

        def update_fct(row_iter):
            point_unique_id = point_model.get_value(
                row_iter,
                self.STORE_COLUMNS.ID
            )
            point = self.target.get_point_by_uid(point_unique_id)
            self._points_store.set_value(
                row_iter,
                self.STORE_COLUMNS.ANGLE,
                point.get_description()
            )

        start_iter = self.get_point_by_uid(uid)
        if point.is_key():
            parent_iter = point_model.iter_parent(start_iter)
            start_iter = parent_iter or start_iter

        self.for_each_in_subtree(
            start_iter,
            update_fct
        )

    def _button_press_cb(self, widget, event):
        single_click = (event.type == Gdk.EventType.BUTTON_PRESS)
        double_click = (event.type == Gdk.EventType._2BUTTON_PRESS)
        is_menu = event.triggers_context_menu()
        x, y = int(event.x), int(event.y)
        bw_x, bw_y = widget.convert_widget_to_bin_window_coords(x, y)
        click_info = widget.get_path_at_pos(bw_x, bw_y)
        if click_info is None:
            return False
        click_treepath, click_col, cell_x, cell_y = click_info
        point_model = self._points_treeview.get_model()
        point_iter = point_model.get_iter(click_treepath)
        point_unique_id = point_model.get_value(
            point_iter,
            self.STORE_COLUMNS.ID
        )

        if is_menu:
            time = event.time
            button = event.button
            self.point_list_menu.popup(None, None, None, None, button, time)
            return False
        elif single_click:
            if click_col is self._locked_column:
                self.target.lock_point(point_unique_id)
                self._touch_subtree(point_iter)
            elif click_col is self._active_column:
                self.target.toggle_point(point_unique_id)
                self._touch_subtree(point_iter)
            elif click_col is self._color_column:
                point = self.target.get_point_by_uid(point_unique_id)
                if not point.is_point():
                    return False
                old_color = self._points_store.get_value(
                    point_iter,
                    self.STORE_COLUMNS.COLOR
                )
                old_color = (
                    (old_color >> 24) / 255.0,          # R
                    ((old_color >> 16) & 255) / 255.0,  # G
                    ((old_color >> 8) & 255) / 255.0,   # B
                )
                color = gui.dialogs.ask_for_color(
                    title=_("Color details"),
                    color=RGBColor(rgb=old_color),
                    previous_color=RGBColor(rgb=old_color),
                )
                if color:
                    point.color = color.to_fill_pixel()
                    self._points_store.set_value(
                        point_iter,
                        self.STORE_COLUMNS.COLOR,
                        point.color
                    )
            else:
                return False
        elif double_click:
            if click_col is self._locked_column:
                pass
            elif click_col is self._active_column:
                pass
            else:
                self._point_edit_action()
        else:
            return False
        return True

    def _add_group_action_cb(self, action, value):
        self.target.create_group()

    def _2P_action_cb(self, action, value):
        self.target.create_space()

    def _3P_action_cb(self, action, value):
        self._add_3p_window.show()

        builder = self._builder
        x_field = builder.get_object("3P_axis_x_field")
        x_field.set_value(0)
        y_field = builder.get_object("3P_axis_y_field")
        y_field.set_value(1)
        z_field = builder.get_object("3P_axis_z_field")
        z_field.set_value(0)
        angle_field = builder.get_object("3P_axis_angle_field")
        angle_field.set_value(0)

    def _create_MP_action_cb(self, action, value):
        """ Create measure points button callback """
        (uid, _) = self.get_selected_point()
        if uid is None:
            return
        self.target.create_compute_point(
            [uid], 'compute_measure_points', [
                {'suffix': 'L', 'additional': (1,)},
                {'suffix': 'R', 'additional': (-1,)}
            ]
        )

    def _create_MP_2_action_cb(self, action, value):
        """ Create measure points (from 2 VPs) button callback """
        points = self.get_selected_points()
        if len(points) != 2:
            return

        uids = []
        for point in points:
            uids.append(point[0])
        self.target.create_compute_point(
            uids, 'compute_measure_points_2', [
                {'suffix': 'L', 'additional': (1,), 'separator': ', '},
                {'suffix': 'R', 'additional': (-1,), 'separator': ', '}
            ]
        )

    def _create_mirrored_action_cb(self, action, value):
        """ Create mirrored points button callback """
        (uid, _) = self.get_selected_point()
        if uid is None:
            return
        self.target.create_compute_point(
            [uid], 'compute_mirrored_points', [{'suffix': 'mirrored'}]
        )

    def _create_cross_product_action_cb(self, action, value):
        """ Create cross product button callback """
        points = self.get_selected_points()
        if len(points) != 2:
            return

        uids = []
        for point in points:
            uids.append(point[0])
        self.target.create_compute_point(
            uids, 'cross_product', [{'separator': ' × '}]
        )

    def _create_2d_direction_action_cb(self, action, value):
        """ Create 2D direction button callback """
        points = self.get_selected_points()
        if len(points) != 2:
            return

        uids = []
        for point in points:
            uids.append(point[0])
        self.target.create_compute_point(
            uids, '2d_direction', [{'suffix': 'dir', 'separator': ', '}]
        )

    def _create_2d_direction_90_action_cb(self, action, value):
        """ Create 2D direction 90° rotated button callback """
        points = self.get_selected_points()
        if len(points) != 2:
            return

        uids = []
        for point in points:
            uids.append(point[0])
        self.target.create_compute_point(
            uids, '2d_direction_90', [{'suffix': 'dir 90°', 'separator': ', '}]
        )

    def _3P_edit_apply_button_cb(self, button):
        builder = self._builder
        x_field = builder.get_object("3P_axis_x_field")
        x = x_field.get_value()
        y_field = builder.get_object("3P_axis_y_field")
        y = y_field.get_value()
        z_field = builder.get_object("3P_axis_z_field")
        z = z_field.get_value()
        angle_field = builder.get_object("3P_axis_angle_field")
        angle = angle_field.get_value()

        self.target.create_space(axis=(x, y, z), angle=angle, name=_("3P"))

        return self._hide_window(self._add_3p_window)

    def _get_selected_preset_name(self):
        preset_iter = self._preset_combo_box.get_active_iter()
        return self._preset_store.get_value(preset_iter, 0)

    def _reset_button_cb(self, button):
        preset_name = self._get_selected_preset_name()
        self.target.reset_preset(preset_name)

    def _add_preset_button_cb(self, button):
        preset_name = self._get_selected_preset_name()
        self.target.add_preset(preset_name)

    def _set_line_tool_cb(self, button):
        self.target._state.set_tool_line()

    def _set_circle_tool_cb(self, button):
        self.target._state.set_tool_circle()

    def _set_polygon_tool_cb(self, button):
        self.target._state.set_tool_polygon()

    def _polygon_side_count_change_cb(self, spin):
        self.target.set_polygon_side_count(spin.get_value())
