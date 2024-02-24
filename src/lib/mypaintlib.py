# This file was automatically generated by SWIG (https://www.swig.org).
# Version 4.1.1
#
# Do not make changes to this file unless you know what you are doing - modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _mypaintlib
else:
    import _mypaintlib

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "this":
            set(self, name, value)
        elif name == "thisown":
            self.this.own(value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _mypaintlib.delete_SwigPyIterator

    def value(self):
        return _mypaintlib.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _mypaintlib.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _mypaintlib.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _mypaintlib.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _mypaintlib.SwigPyIterator_equal(self, x)

    def copy(self):
        return _mypaintlib.SwigPyIterator_copy(self)

    def next(self):
        return _mypaintlib.SwigPyIterator_next(self)

    def __next__(self):
        return _mypaintlib.SwigPyIterator___next__(self)

    def previous(self):
        return _mypaintlib.SwigPyIterator_previous(self)

    def advance(self, n):
        return _mypaintlib.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _mypaintlib.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _mypaintlib.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _mypaintlib.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _mypaintlib.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _mypaintlib.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _mypaintlib.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _mypaintlib:
_mypaintlib.SwigPyIterator_swigregister(SwigPyIterator)
cvar = _mypaintlib.cvar
heavy_debug = cvar.heavy_debug

class IntVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _mypaintlib.IntVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _mypaintlib.IntVector___nonzero__(self)

    def __bool__(self):
        return _mypaintlib.IntVector___bool__(self)

    def __len__(self):
        return _mypaintlib.IntVector___len__(self)

    def __getslice__(self, i, j):
        return _mypaintlib.IntVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _mypaintlib.IntVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _mypaintlib.IntVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _mypaintlib.IntVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _mypaintlib.IntVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _mypaintlib.IntVector___setitem__(self, *args)

    def pop(self):
        return _mypaintlib.IntVector_pop(self)

    def append(self, x):
        return _mypaintlib.IntVector_append(self, x)

    def empty(self):
        return _mypaintlib.IntVector_empty(self)

    def size(self):
        return _mypaintlib.IntVector_size(self)

    def swap(self, v):
        return _mypaintlib.IntVector_swap(self, v)

    def begin(self):
        return _mypaintlib.IntVector_begin(self)

    def end(self):
        return _mypaintlib.IntVector_end(self)

    def rbegin(self):
        return _mypaintlib.IntVector_rbegin(self)

    def rend(self):
        return _mypaintlib.IntVector_rend(self)

    def clear(self):
        return _mypaintlib.IntVector_clear(self)

    def get_allocator(self):
        return _mypaintlib.IntVector_get_allocator(self)

    def pop_back(self):
        return _mypaintlib.IntVector_pop_back(self)

    def erase(self, *args):
        return _mypaintlib.IntVector_erase(self, *args)

    def __init__(self, *args):
        _mypaintlib.IntVector_swiginit(self, _mypaintlib.new_IntVector(*args))

    def push_back(self, x):
        return _mypaintlib.IntVector_push_back(self, x)

    def front(self):
        return _mypaintlib.IntVector_front(self)

    def back(self):
        return _mypaintlib.IntVector_back(self)

    def assign(self, n, x):
        return _mypaintlib.IntVector_assign(self, n, x)

    def resize(self, *args):
        return _mypaintlib.IntVector_resize(self, *args)

    def insert(self, *args):
        return _mypaintlib.IntVector_insert(self, *args)

    def reserve(self, n):
        return _mypaintlib.IntVector_reserve(self, n)

    def capacity(self):
        return _mypaintlib.IntVector_capacity(self)
    __swig_destroy__ = _mypaintlib.delete_IntVector

# Register IntVector in _mypaintlib:
_mypaintlib.IntVector_swigregister(IntVector)
class RectVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _mypaintlib.RectVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _mypaintlib.RectVector___nonzero__(self)

    def __bool__(self):
        return _mypaintlib.RectVector___bool__(self)

    def __len__(self):
        return _mypaintlib.RectVector___len__(self)

    def __getslice__(self, i, j):
        return _mypaintlib.RectVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _mypaintlib.RectVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _mypaintlib.RectVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _mypaintlib.RectVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _mypaintlib.RectVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _mypaintlib.RectVector___setitem__(self, *args)

    def pop(self):
        return _mypaintlib.RectVector_pop(self)

    def append(self, x):
        return _mypaintlib.RectVector_append(self, x)

    def empty(self):
        return _mypaintlib.RectVector_empty(self)

    def size(self):
        return _mypaintlib.RectVector_size(self)

    def swap(self, v):
        return _mypaintlib.RectVector_swap(self, v)

    def begin(self):
        return _mypaintlib.RectVector_begin(self)

    def end(self):
        return _mypaintlib.RectVector_end(self)

    def rbegin(self):
        return _mypaintlib.RectVector_rbegin(self)

    def rend(self):
        return _mypaintlib.RectVector_rend(self)

    def clear(self):
        return _mypaintlib.RectVector_clear(self)

    def get_allocator(self):
        return _mypaintlib.RectVector_get_allocator(self)

    def pop_back(self):
        return _mypaintlib.RectVector_pop_back(self)

    def erase(self, *args):
        return _mypaintlib.RectVector_erase(self, *args)

    def __init__(self, *args):
        _mypaintlib.RectVector_swiginit(self, _mypaintlib.new_RectVector(*args))

    def push_back(self, x):
        return _mypaintlib.RectVector_push_back(self, x)

    def front(self):
        return _mypaintlib.RectVector_front(self)

    def back(self):
        return _mypaintlib.RectVector_back(self)

    def assign(self, n, x):
        return _mypaintlib.RectVector_assign(self, n, x)

    def resize(self, *args):
        return _mypaintlib.RectVector_resize(self, *args)

    def insert(self, *args):
        return _mypaintlib.RectVector_insert(self, *args)

    def reserve(self, n):
        return _mypaintlib.RectVector_reserve(self, n)

    def capacity(self):
        return _mypaintlib.RectVector_capacity(self)
    __swig_destroy__ = _mypaintlib.delete_RectVector

# Register RectVector in _mypaintlib:
_mypaintlib.RectVector_swigregister(RectVector)
class DoubleVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _mypaintlib.DoubleVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _mypaintlib.DoubleVector___nonzero__(self)

    def __bool__(self):
        return _mypaintlib.DoubleVector___bool__(self)

    def __len__(self):
        return _mypaintlib.DoubleVector___len__(self)

    def __getslice__(self, i, j):
        return _mypaintlib.DoubleVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _mypaintlib.DoubleVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _mypaintlib.DoubleVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _mypaintlib.DoubleVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _mypaintlib.DoubleVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _mypaintlib.DoubleVector___setitem__(self, *args)

    def pop(self):
        return _mypaintlib.DoubleVector_pop(self)

    def append(self, x):
        return _mypaintlib.DoubleVector_append(self, x)

    def empty(self):
        return _mypaintlib.DoubleVector_empty(self)

    def size(self):
        return _mypaintlib.DoubleVector_size(self)

    def swap(self, v):
        return _mypaintlib.DoubleVector_swap(self, v)

    def begin(self):
        return _mypaintlib.DoubleVector_begin(self)

    def end(self):
        return _mypaintlib.DoubleVector_end(self)

    def rbegin(self):
        return _mypaintlib.DoubleVector_rbegin(self)

    def rend(self):
        return _mypaintlib.DoubleVector_rend(self)

    def clear(self):
        return _mypaintlib.DoubleVector_clear(self)

    def get_allocator(self):
        return _mypaintlib.DoubleVector_get_allocator(self)

    def pop_back(self):
        return _mypaintlib.DoubleVector_pop_back(self)

    def erase(self, *args):
        return _mypaintlib.DoubleVector_erase(self, *args)

    def __init__(self, *args):
        _mypaintlib.DoubleVector_swiginit(self, _mypaintlib.new_DoubleVector(*args))

    def push_back(self, x):
        return _mypaintlib.DoubleVector_push_back(self, x)

    def front(self):
        return _mypaintlib.DoubleVector_front(self)

    def back(self):
        return _mypaintlib.DoubleVector_back(self)

    def assign(self, n, x):
        return _mypaintlib.DoubleVector_assign(self, n, x)

    def resize(self, *args):
        return _mypaintlib.DoubleVector_resize(self, *args)

    def insert(self, *args):
        return _mypaintlib.DoubleVector_insert(self, *args)

    def reserve(self, n):
        return _mypaintlib.DoubleVector_reserve(self, n)

    def capacity(self):
        return _mypaintlib.DoubleVector_capacity(self)
    __swig_destroy__ = _mypaintlib.delete_DoubleVector

# Register DoubleVector in _mypaintlib:
_mypaintlib.DoubleVector_swigregister(DoubleVector)
class Rect(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    x = property(_mypaintlib.Rect_x_get, _mypaintlib.Rect_x_set)
    y = property(_mypaintlib.Rect_y_get, _mypaintlib.Rect_y_set)
    w = property(_mypaintlib.Rect_w_get, _mypaintlib.Rect_w_set)
    h = property(_mypaintlib.Rect_h_get, _mypaintlib.Rect_h_set)

    def __init__(self):
        _mypaintlib.Rect_swiginit(self, _mypaintlib.new_Rect())
    __swig_destroy__ = _mypaintlib.delete_Rect

# Register Rect in _mypaintlib:
_mypaintlib.Rect_swigregister(Rect)
class Surface(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _mypaintlib.delete_Surface

    def get_surface_interface(self):
        return _mypaintlib.Surface_get_surface_interface(self)

# Register Surface in _mypaintlib:
_mypaintlib.Surface_swigregister(Surface)
class Brush(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _mypaintlib.Brush_swiginit(self, _mypaintlib.new_Brush())
    __swig_destroy__ = _mypaintlib.delete_Brush

    def reset(self):
        return _mypaintlib.Brush_reset(self)

    def new_stroke(self):
        return _mypaintlib.Brush_new_stroke(self)

    def set_base_value(self, id, value):
        return _mypaintlib.Brush_set_base_value(self, id, value)

    def set_mapping_n(self, id, input, n):
        return _mypaintlib.Brush_set_mapping_n(self, id, input, n)

    def set_mapping_point(self, id, input, index, x, y):
        return _mypaintlib.Brush_set_mapping_point(self, id, input, index, x, y)

    def get_state(self, i):
        return _mypaintlib.Brush_get_state(self, i)

    def set_state(self, i, value):
        return _mypaintlib.Brush_set_state(self, i, value)

    def stroke_to(self, surface, x, y, pressure, xtilt, ytilt, dtime, viewzoom, viewrotation, barrel_rotation, linear):
        return _mypaintlib.Brush_stroke_to(self, surface, x, y, pressure, xtilt, ytilt, dtime, viewzoom, viewrotation, barrel_rotation, linear)

    def get_total_stroke_painting_time(self):
        return _mypaintlib.Brush_get_total_stroke_painting_time(self)

    def set_print_inputs(self, enabled):
        return _mypaintlib.Brush_set_print_inputs(self, enabled)

# Register Brush in _mypaintlib:
_mypaintlib.Brush_swigregister(Brush)
class MappingWrapper(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, inputs_):
        _mypaintlib.MappingWrapper_swiginit(self, _mypaintlib.new_MappingWrapper(inputs_))
    __swig_destroy__ = _mypaintlib.delete_MappingWrapper

    def set_n(self, input, n):
        return _mypaintlib.MappingWrapper_set_n(self, input, n)

    def set_point(self, input, index, x, y):
        return _mypaintlib.MappingWrapper_set_point(self, input, index, x, y)

    def is_constant(self):
        return _mypaintlib.MappingWrapper_is_constant(self)

    def calculate(self, data):
        return _mypaintlib.MappingWrapper_calculate(self, data)

    def calculate_single_input(self, input):
        return _mypaintlib.MappingWrapper_calculate_single_input(self, input)

# Register MappingWrapper in _mypaintlib:
_mypaintlib.MappingWrapper_swigregister(MappingWrapper)
class PythonBrush(Brush):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def get_states_as_array(self):
        return _mypaintlib.PythonBrush_get_states_as_array(self)

    def set_states_from_array(self, obj):
        return _mypaintlib.PythonBrush_set_states_from_array(self, obj)

    def stroke_to(self, surface, x, y, pressure, xtilt, ytilt, dtime, viewzoom, viewrotation, barrel_rotation, linear):
        return _mypaintlib.PythonBrush_stroke_to(self, surface, x, y, pressure, xtilt, ytilt, dtime, viewzoom, viewrotation, barrel_rotation, linear)

    def __init__(self):
        _mypaintlib.PythonBrush_swiginit(self, _mypaintlib.new_PythonBrush())
    __swig_destroy__ = _mypaintlib.delete_PythonBrush

# Register PythonBrush in _mypaintlib:
_mypaintlib.PythonBrush_swigregister(PythonBrush)
BBOXES = _mypaintlib.BBOXES
SymmetryVertical = _mypaintlib.SymmetryVertical
SymmetryHorizontal = _mypaintlib.SymmetryHorizontal
SymmetryVertHorz = _mypaintlib.SymmetryVertHorz
SymmetryRotational = _mypaintlib.SymmetryRotational
SymmetrySnowflake = _mypaintlib.SymmetrySnowflake
NumSymmetryTypes = _mypaintlib.NumSymmetryTypes
class TiledSurface(Surface):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, self_):
        _mypaintlib.TiledSurface_swiginit(self, _mypaintlib.new_TiledSurface(self_))
    __swig_destroy__ = _mypaintlib.delete_TiledSurface

    def set_symmetry_state(self, active, center_x, center_y, symmetry_type, rot_symmetry_lines, symmetry_angle):
        return _mypaintlib.TiledSurface_set_symmetry_state(self, active, center_x, center_y, symmetry_type, rot_symmetry_lines, symmetry_angle)

    def begin_atomic(self):
        return _mypaintlib.TiledSurface_begin_atomic(self)

    def end_atomic(self):
        return _mypaintlib.TiledSurface_end_atomic(self)

    def draw_dab(self, x, y, radius, color_r, color_g, color_b, opaque, hardness=0.5, softness=0.0, color_a=1.0, aspect_ratio=1.0, angle=0.0, lock_alpha=0.0, colorize=0.0, posterize=0.0, posterize_num=0.0, paint=1.0):
        return _mypaintlib.TiledSurface_draw_dab(self, x, y, radius, color_r, color_g, color_b, opaque, hardness, softness, color_a, aspect_ratio, angle, lock_alpha, colorize, posterize, posterize_num, paint)

    def get_color(self, x, y, radius):
        return _mypaintlib.TiledSurface_get_color(self, x, y, radius)

    def get_alpha(self, x, y, radius):
        return _mypaintlib.TiledSurface_get_alpha(self, x, y, radius)

    def get_surface_interface(self):
        return _mypaintlib.TiledSurface_get_surface_interface(self)

# Register TiledSurface in _mypaintlib:
_mypaintlib.TiledSurface_swigregister(TiledSurface)
TILE_SIZE = cvar.TILE_SIZE
MAX_MIPMAP_LEVEL = cvar.MAX_MIPMAP_LEVEL


def get_module(name):
    return _mypaintlib.get_module(name)

def new_py_tiled_surface(pModule):
    return _mypaintlib.new_py_tiled_surface(pModule)

def mypaint_python_surface_factory(user_data):
    return _mypaintlib.mypaint_python_surface_factory(user_data)

def tile_downscale_rgba16(src, dst, dst_x, dst_y):
    return _mypaintlib.tile_downscale_rgba16(src, dst, dst_x, dst_y)

def tile_copy_rgba16_into_rgba16(src, dst):
    return _mypaintlib.tile_copy_rgba16_into_rgba16(src, dst)

def tile_clear_rgba16(dst):
    return _mypaintlib.tile_clear_rgba16(dst)

def tile_clear_rgba8(dst):
    return _mypaintlib.tile_clear_rgba8(dst)

def tile_convert_rgba16_to_rgba8(src, dst, EOTF):
    return _mypaintlib.tile_convert_rgba16_to_rgba8(src, dst, EOTF)

def tile_convert_rgbu16_to_rgbu8(src, dst, EOTF):
    return _mypaintlib.tile_convert_rgbu16_to_rgbu8(src, dst, EOTF)

def tile_convert_rgba8_to_rgba16(src, dst, EOTF):
    return _mypaintlib.tile_convert_rgba8_to_rgba16(src, dst, EOTF)

def tile_rgba2flat(dst_obj, bg_obj):
    return _mypaintlib.tile_rgba2flat(dst_obj, bg_obj)

def tile_flat2rgba(dst_obj, bg_obj):
    return _mypaintlib.tile_flat2rgba(dst_obj, bg_obj)

def tile_perceptual_change_strokemap(a_obj, b_obj, res_obj):
    return _mypaintlib.tile_perceptual_change_strokemap(a_obj, b_obj, res_obj)
CombineNormal = _mypaintlib.CombineNormal
CombineMultiply = _mypaintlib.CombineMultiply
CombineScreen = _mypaintlib.CombineScreen
CombineOverlay = _mypaintlib.CombineOverlay
CombineDarken = _mypaintlib.CombineDarken
CombineLighten = _mypaintlib.CombineLighten
CombineHardLight = _mypaintlib.CombineHardLight
CombineSoftLight = _mypaintlib.CombineSoftLight
CombineColorBurn = _mypaintlib.CombineColorBurn
CombineColorDodge = _mypaintlib.CombineColorDodge
CombineDifference = _mypaintlib.CombineDifference
CombineExclusion = _mypaintlib.CombineExclusion
CombineHue = _mypaintlib.CombineHue
CombineSaturation = _mypaintlib.CombineSaturation
CombineColor = _mypaintlib.CombineColor
CombineLuminosity = _mypaintlib.CombineLuminosity
CombineLighter = _mypaintlib.CombineLighter
CombineDestinationIn = _mypaintlib.CombineDestinationIn
CombineDestinationOut = _mypaintlib.CombineDestinationOut
CombineSourceAtop = _mypaintlib.CombineSourceAtop
CombineDestinationAtop = _mypaintlib.CombineDestinationAtop
CombineSpectralWGM = _mypaintlib.CombineSpectralWGM
NumCombineModes = _mypaintlib.NumCombineModes

def combine_mode_get_info(mode):
    return _mypaintlib.combine_mode_get_info(mode)

def tile_combine(mode, src_obj, dst_obj, dst_has_alpha, src_opacity):
    return _mypaintlib.tile_combine(mode, src_obj, dst_obj, dst_has_alpha, src_opacity)
class SCWSColorSelector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def get_size(self):
        return _mypaintlib.SCWSColorSelector_get_size(self)
    brush_h = property(_mypaintlib.SCWSColorSelector_brush_h_get, _mypaintlib.SCWSColorSelector_brush_h_set)
    brush_s = property(_mypaintlib.SCWSColorSelector_brush_s_get, _mypaintlib.SCWSColorSelector_brush_s_set)
    brush_v = property(_mypaintlib.SCWSColorSelector_brush_v_get, _mypaintlib.SCWSColorSelector_brush_v_set)

    def set_brush_color(self, h, s, v):
        return _mypaintlib.SCWSColorSelector_set_brush_color(self, h, s, v)

    def get_hsva_at(self, h, s, v, a, x, y, adjust_color=True, only_colors=True, mark_h=0.0):
        return _mypaintlib.SCWSColorSelector_get_hsva_at(self, h, s, v, a, x, y, adjust_color, only_colors, mark_h)

    def pick_color_at(self, x, y):
        return _mypaintlib.SCWSColorSelector_pick_color_at(self, x, y)

    def render(self, obj):
        return _mypaintlib.SCWSColorSelector_render(self, obj)

    def __init__(self):
        _mypaintlib.SCWSColorSelector_swiginit(self, _mypaintlib.new_SCWSColorSelector())
    __swig_destroy__ = _mypaintlib.delete_SCWSColorSelector

# Register SCWSColorSelector in _mypaintlib:
_mypaintlib.SCWSColorSelector_swigregister(SCWSColorSelector)
colorring_size = cvar.colorring_size
center = cvar.center
RAD_TO_ONE = cvar.RAD_TO_ONE
TWO_PI = cvar.TWO_PI
ONE_OVER_THREE = cvar.ONE_OVER_THREE
TWO_OVER_THREE = cvar.TWO_OVER_THREE

class ColorChangerWash(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    brush_h = property(_mypaintlib.ColorChangerWash_brush_h_get, _mypaintlib.ColorChangerWash_brush_h_set)
    brush_s = property(_mypaintlib.ColorChangerWash_brush_s_get, _mypaintlib.ColorChangerWash_brush_s_set)
    brush_v = property(_mypaintlib.ColorChangerWash_brush_v_get, _mypaintlib.ColorChangerWash_brush_v_set)

    def set_brush_color(self, h, s, v):
        return _mypaintlib.ColorChangerWash_set_brush_color(self, h, s, v)

    def get_size(self):
        return _mypaintlib.ColorChangerWash_get_size(self)

    def render(self, obj):
        return _mypaintlib.ColorChangerWash_render(self, obj)

    def pick_color_at(self, x_, y_):
        return _mypaintlib.ColorChangerWash_pick_color_at(self, x_, y_)

    def __init__(self):
        _mypaintlib.ColorChangerWash_swiginit(self, _mypaintlib.new_ColorChangerWash())
    __swig_destroy__ = _mypaintlib.delete_ColorChangerWash

# Register ColorChangerWash in _mypaintlib:
_mypaintlib.ColorChangerWash_swigregister(ColorChangerWash)
ccw_size = cvar.ccw_size
v06_colorchanger = cvar.v06_colorchanger

class ColorChangerCrossedBowl(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    brush_h = property(_mypaintlib.ColorChangerCrossedBowl_brush_h_get, _mypaintlib.ColorChangerCrossedBowl_brush_h_set)
    brush_s = property(_mypaintlib.ColorChangerCrossedBowl_brush_s_get, _mypaintlib.ColorChangerCrossedBowl_brush_s_set)
    brush_v = property(_mypaintlib.ColorChangerCrossedBowl_brush_v_get, _mypaintlib.ColorChangerCrossedBowl_brush_v_set)

    def set_brush_color(self, h, s, v):
        return _mypaintlib.ColorChangerCrossedBowl_set_brush_color(self, h, s, v)

    def get_size(self):
        return _mypaintlib.ColorChangerCrossedBowl_get_size(self)

    def render(self, obj):
        return _mypaintlib.ColorChangerCrossedBowl_render(self, obj)

    def pick_color_at(self, x_, y_):
        return _mypaintlib.ColorChangerCrossedBowl_pick_color_at(self, x_, y_)

    def __init__(self):
        _mypaintlib.ColorChangerCrossedBowl_swiginit(self, _mypaintlib.new_ColorChangerCrossedBowl())
    __swig_destroy__ = _mypaintlib.delete_ColorChangerCrossedBowl

# Register ColorChangerCrossedBowl in _mypaintlib:
_mypaintlib.ColorChangerCrossedBowl_swigregister(ColorChangerCrossedBowl)
ccdb_size = cvar.ccdb_size

class ProgressivePNGWriter(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, file, w, h, has_alpha, save_srgb_chunks):
        _mypaintlib.ProgressivePNGWriter_swiginit(self, _mypaintlib.new_ProgressivePNGWriter(file, w, h, has_alpha, save_srgb_chunks))

    def write(self, arr):
        return _mypaintlib.ProgressivePNGWriter_write(self, arr)

    def close(self):
        return _mypaintlib.ProgressivePNGWriter_close(self)
    __swig_destroy__ = _mypaintlib.delete_ProgressivePNGWriter

# Register ProgressivePNGWriter in _mypaintlib:
_mypaintlib.ProgressivePNGWriter_swigregister(ProgressivePNGWriter)

def load_png_fast_progressive(filename, get_buffer_callback, convert_to_srgb):
    return _mypaintlib.load_png_fast_progressive(filename, get_buffer_callback, convert_to_srgb)
class ConstTiles(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    @staticmethod
    def ALPHA_OPAQUE():
        return _mypaintlib.ConstTiles_ALPHA_OPAQUE()

    @staticmethod
    def ALPHA_TRANSPARENT():
        return _mypaintlib.ConstTiles_ALPHA_TRANSPARENT()

    def __init__(self):
        _mypaintlib.ConstTiles_swiginit(self, _mypaintlib.new_ConstTiles())
    __swig_destroy__ = _mypaintlib.delete_ConstTiles

# Register ConstTiles in _mypaintlib:
_mypaintlib.ConstTiles_swigregister(ConstTiles)
class edges(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    north = _mypaintlib.edges_north
    east = _mypaintlib.edges_east
    south = _mypaintlib.edges_south
    west = _mypaintlib.edges_west
    none = _mypaintlib.edges_none

    def __init__(self):
        _mypaintlib.edges_swiginit(self, _mypaintlib.new_edges())
    __swig_destroy__ = _mypaintlib.delete_edges

# Register edges in _mypaintlib:
_mypaintlib.edges_swigregister(edges)
class Filler(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, targ_r, targ_g, targ_b, targ_a, tol):
        _mypaintlib.Filler_swiginit(self, _mypaintlib.new_Filler(targ_r, targ_g, targ_b, targ_a, tol))

    def fill(self, src, dst, seeds, direction, min_x, min_y, max_x, max_y):
        return _mypaintlib.Filler_fill(self, src, dst, seeds, direction, min_x, min_y, max_x, max_y)

    def flood(self, src, dst):
        return _mypaintlib.Filler_flood(self, src, dst)

    def tile_uniformity(self, is_empty, src):
        return _mypaintlib.Filler_tile_uniformity(self, is_empty, src)
    __swig_destroy__ = _mypaintlib.delete_Filler

# Register Filler in _mypaintlib:
_mypaintlib.Filler_swigregister(Filler)
class GapClosingFiller(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, max_dist, track_seep):
        _mypaintlib.GapClosingFiller_swiginit(self, _mypaintlib.new_GapClosingFiller(max_dist, track_seep))

    def fill(self, alphas, distances, dst, seeds, min_x, min_y, max_x, max_y):
        return _mypaintlib.GapClosingFiller_fill(self, alphas, distances, dst, seeds, min_x, min_y, max_x, max_y)

    def unseep(self, distances, dst, seeds, initial):
        return _mypaintlib.GapClosingFiller_unseep(self, distances, dst, seeds, initial)
    __swig_destroy__ = _mypaintlib.delete_GapClosingFiller

# Register GapClosingFiller in _mypaintlib:
_mypaintlib.GapClosingFiller_swigregister(GapClosingFiller)

def rgba_tile_from_alpha_tile(src, fill_r, fill_g, fill_b, min_x, min_y, max_x, max_y):
    return _mypaintlib.rgba_tile_from_alpha_tile(src, fill_r, fill_g, fill_b, min_x, min_y, max_x, max_y)
class DistanceBucket(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, distance):
        _mypaintlib.DistanceBucket_swiginit(self, _mypaintlib.new_DistanceBucket(distance))
    __swig_destroy__ = _mypaintlib.delete_DistanceBucket
    distance = property(_mypaintlib.DistanceBucket_distance_get)
    input = property(_mypaintlib.DistanceBucket_input_get, _mypaintlib.DistanceBucket_input_set)

# Register DistanceBucket in _mypaintlib:
_mypaintlib.DistanceBucket_swigregister(DistanceBucket)

def find_gaps(bucket, gap_output, src_mid, src_n, src_e, src_s, src_w, src_ne, src_se, src_sw, src_nw):
    return _mypaintlib.find_gaps(bucket, gap_output, src_mid, src_n, src_e, src_s, src_w, src_ne, src_se, src_sw, src_nw)
class Controller(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _mypaintlib.Controller_swiginit(self, _mypaintlib.new_Controller())

    def stop(self):
        return _mypaintlib.Controller_stop(self)

    def inc_processed(self, incr):
        return _mypaintlib.Controller_inc_processed(self, incr)

    def num_processed(self):
        return _mypaintlib.Controller_num_processed(self)

    def reset(self):
        return _mypaintlib.Controller_reset(self)
    __swig_destroy__ = _mypaintlib.delete_Controller

# Register Controller in _mypaintlib:
_mypaintlib.Controller_swigregister(Controller)

def morph(offset, morphed, tiles, strands, status_controller):
    return _mypaintlib.morph(offset, morphed, tiles, strands, status_controller)

def blur(radius, blurred, tiles, strands, status_controller):
    return _mypaintlib.blur(radius, blurred, tiles, strands, status_controller)

def get_libmypaint_brush_settings():
    return _mypaintlib.get_libmypaint_brush_settings()

def get_libmypaint_brush_inputs():
    return _mypaintlib.get_libmypaint_brush_inputs()

def gdkpixbuf_get_pixels_array(pixbuf_pyobject):
    return _mypaintlib.gdkpixbuf_get_pixels_array(pixbuf_pyobject)

