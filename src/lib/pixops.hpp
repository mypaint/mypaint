/* This file is part of MyPaint.
 * Copyright (C) 2008-2014 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef PIXOPS_HPP
#define PIXOPS_HPP


#include <Python.h>


// Downscales a tile to half its size using bilinear interpolation.  Used for
// generating mipmaps for tiledsurface and background.

void tile_downscale_rgba16(PyObject *src, PyObject *dst, int dst_x, int dst_y);


// Used to e.g. copy the background before starting to composite over it
//
// Simple array copying (numpy assignment operator) is about 13 times slower,
// sadly. The above comment is true when the array is sliced; it's only about
// two times faster now, in the current use case.

void tile_copy_rgba16_into_rgba16(PyObject *src, PyObject *dst);


// Clears a tile.
// This zeroes the alpha channel too, so using it on rgbu data
// may have unexpected consequences.

void tile_clear_rgba16(PyObject *dst);

void tile_clear_rgba8(PyObject *dst);


// Converts a 15ish-bit tile array to 8bpp RGBA.
// Used mainly for saving layers when alpha must be preserved.

void tile_convert_rgba16_to_rgba8(PyObject *src, PyObject *dst, const float EOTF);


// Converts a 15ish-bit tile array to 8bpp RGB ("ignoring" alpha).

void tile_convert_rgbu16_to_rgbu8(PyObject *src, PyObject *dst, const float EOTF);


// used mainly for loading layers (transparent PNG)

void tile_convert_rgba8_to_rgba16(PyObject *src, PyObject *dst, const float EOTF);


// Flatten a premultiplied rgba layer, using "bg" as background.
// (bg is assumed to be flat, bg.alpha is ignored)
//
// dst.color = dst OVER bg.color
// dst.alpha = unmodified

void tile_rgba2flat(PyObject *dst_obj, PyObject *bg_obj);


// Make a flat layer translucent again. When calculating the new color
// and alpha, it is assumed that the layer will be displayed OVER the
// background "bg". Alpha is increased where required.
//
// dst.alpha = MIN(dst.alpha, minimum alpha required for correct result)
// dst.color = calculated such that (dst_output OVER bg = dst_input.color)

void tile_flat2rgba(PyObject * dst_obj, PyObject * bg_obj);


// Calculates a 1-bit bitmap of the stroke shape using two snapshots of the
// layer (the layer before and after the stroke). Used in strokemap.py
//
// If the alpha increases a lot, we want the stroke to appear in the strokemap,
// even if the color did not change. If the alpha decreases a lot, we want to
// ignore the stroke (eraser). If the alpha decreases just a little, but the
// color changes a lot (eg. heavy smudging or watercolor brushes) we want the
// stroke still to be pickable.
//
// If the layer alpha was (near) zero, we record the stroke even if it is
// barely visible. This gives a bigger target to point-and-select.

void tile_perceptual_change_strokemap(PyObject *a_obj, PyObject *b_obj, PyObject *res_obj);


// Tile blending & compositing modes

enum CombineMode {
    CombineNormal,
    CombineMultiply,
    CombineScreen,
    CombineOverlay,
    CombineDarken,
    CombineLighten,   // lightEN blend mode, and Porter-Duff OVER
    CombineHardLight,
    CombineSoftLight,
    CombineColorBurn,
    CombineColorDodge,
    CombineDifference,
    CombineExclusion,
    CombineHue,
    CombineSaturation,
    CombineColor,
    CombineLuminosity,
    CombineLighter,   // normal blend mode, and W3C lightER (Porter-Duff PLUS)
    CombineDestinationIn,
    CombineDestinationOut,
    CombineSourceAtop,
    CombineDestinationAtop,
    CombineSpectralWGM,
    NumCombineModes
};


// Extracts Python-readable metadata for a blend/composite mode

PyObject *
combine_mode_get_info(enum CombineMode mode);


// Blend and composite one tile, writing into the destination.

void
tile_combine (enum CombineMode mode,
              PyObject *src_obj,
              PyObject *dst_obj,
              const bool dst_has_alpha,
              const float src_opacity);


#endif // PIXOPS_HPP
