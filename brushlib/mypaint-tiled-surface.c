/* This file is part of MyPaint.
 * Copyright (C) 2008-2011 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include <math.h>
#include <stdio.h>
#include <assert.h>

#include "mypaint-tiled-surface.h"
#include "helpers.h"

#define TILE_SIZE 64
#define MAX_MIPMAP_LEVEL 4

void mypaint_tiled_surface_begin_atomic(MyPaintTiledSurface *self)
{
    if (self->begin_atomic)
        self->begin_atomic(self);
}

void mypaint_tiled_surface_end_atomic(MyPaintTiledSurface *self)
{
    if (self->end_atomic)
        self->end_atomic(self);
}

uint16_t * mypaint_tiled_surface_get_tile(MyPaintTiledSurface *self, int tx, int ty, gboolean readonly)
{
    if (!self->get_tile)
        return NULL;

    return self->get_tile(self, tx, ty, readonly);
}

void mypaint_tiled_surface_update_tile(MyPaintTiledSurface *self, int tx, int ty, uint16_t * tile_buffer)
{
    if (self->update_tile)
        self->update_tile(self, tx, ty, tile_buffer);
}

void mypaint_tiled_surface_area_changed(MyPaintTiledSurface *self, int bb_x, int bb_y, int bb_w, int bb_h)
{
    if (self->area_changed)
        self->area_changed(self, bb_x, bb_y, bb_w, bb_h);
}

void
mypaint_tiled_surface_set_symmetry_state(MyPaintTiledSurface *self, gboolean active, float center_x)
{
    self->surface_do_symmetry = active;
    self->surface_center_x = center_x;
}

void render_dab_mask (uint16_t * mask,
                        float x, float y,
                        float radius,
                        float hardness,
                        float aspect_ratio, float angle
                        )
{

    hardness = CLAMP(hardness, 0.0, 1.0);
    if (aspect_ratio<1.0) aspect_ratio=1.0;
    assert(hardness != 0.0); // assured by caller

    float r_fringe;
    int xp, yp;
    float xx, yy, rr;
    float one_over_radius2;

    r_fringe = radius + 1;
    rr = radius*radius;
    one_over_radius2 = 1.0/rr;

    // For a graphical explanation, see:
    // http://wiki.mypaint.info/Development/Documentation/Brushlib
    //
    // The hardness calculation is explained below:
    //
    // Dab opacity gradually fades out from the center (rr=0) to
    // fringe (rr=1) of the dab. How exactly depends on the hardness.
    // We use two linear segments, for which we pre-calculate slope
    // and offset here.
    //
    // opa
    // ^
    // *   .
    // |        *
    // |          .
    // +-----------*> rr = (distance_from_center/radius)^2
    // 0           1
    //
    float segment1_offset = 1.0;
    float segment1_slope  = -(1.0/hardness - 1.0);
    float segment2_offset = hardness/(1.0-hardness);
    float segment2_slope  = -hardness/(1.0-hardness);
    // for hardness == 1.0, segment2 will never be used

    float angle_rad=angle/360*2*M_PI;
    float cs=cos(angle_rad);
    float sn=sin(angle_rad);

    int x0 = floor (x - r_fringe);
    int y0 = floor (y - r_fringe);
    int x1 = ceil (x + r_fringe);
    int y1 = ceil (y + r_fringe);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > TILE_SIZE-1) x1 = TILE_SIZE-1;
    if (y1 > TILE_SIZE-1) y1 = TILE_SIZE-1;


    // we do run length encoding: if opacity is zero, the next
    // value in the mask is the number of pixels that can be skipped.
    uint16_t * mask_p = mask;
    int skip=0;

    skip += y0*TILE_SIZE;
    for (yp = y0; yp <= y1; yp++) {
      yy = (yp + 0.5 - y);
      skip += x0;
      for (xp = x0; xp <= x1; xp++) {
        xx = (xp + 0.5 - x);
        // code duplication, see brush::count_dabs_to()
        float yyr=(yy*cs-xx*sn)*aspect_ratio;
        float xxr=yy*sn+xx*cs;
        rr = (yyr*yyr + xxr*xxr) * one_over_radius2;
        // rr is in range 0.0..1.0*sqrt(2)

        float opa;
        if (rr <= 1.0) {
          float fac;
          if (rr <= hardness) {
            opa = segment1_offset;
            fac = segment1_slope;
          } else {
            opa = segment2_offset;
            fac = segment2_slope;
          }
          opa += rr*fac;

#ifdef HEAVY_DEBUG
          assert(isfinite(opa));
          assert(opa >= 0.0 && opa <= 1.0);
#endif
        } else {
          opa = 0.0;
        }

        uint16_t opa_ = opa * (1<<15);
        if (!opa_) {
          skip++;
        } else {
          if (skip) {
            *mask_p++ = 0;
            *mask_p++ = skip*4;
            skip = 0;
          }
          *mask_p++ = opa_;
        }
      }
      skip += TILE_SIZE-xp;
    }
    *mask_p++ = 0;
    *mask_p++ = 0;
  }

// returns TRUE if the surface was modified
gboolean draw_dab_internal (MyPaintTiledSurface *self, float x, float y,
               float radius,
               float color_r, float color_g, float color_b,
               float opaque, float hardness,
               float color_a,
               float aspect_ratio, float angle,
               float lock_alpha,
               float colorize,
               int recursing // used for symmetry, internal use only
               )

{
    opaque = CLAMP(opaque, 0.0, 1.0);
    hardness = CLAMP(hardness, 0.0, 1.0);
    lock_alpha = CLAMP(lock_alpha, 0.0, 1.0);
    colorize = CLAMP(colorize, 0.0, 1.0);
    if (radius < 0.1) return FALSE; // don't bother with dabs smaller than 0.1 pixel
    if (hardness == 0.0) return FALSE; // infintly small center point, fully transparent outside
    if (opaque == 0.0) return FALSE;

    color_r = CLAMP(color_r, 0.0, 1.0);
    color_g = CLAMP(color_g, 0.0, 1.0);
    color_b = CLAMP(color_b, 0.0, 1.0);
    color_a = CLAMP(color_a, 0.0, 1.0);

    uint16_t color_r_ = color_r * (1<<15);
    uint16_t color_g_ = color_g * (1<<15);
    uint16_t color_b_ = color_b * (1<<15);

    // blending mode preparation
    float normal = 1.0;

    normal *= 1.0-lock_alpha;
    normal *= 1.0-colorize;

    if (aspect_ratio<1.0) aspect_ratio=1.0;

    float r_fringe = radius + 1;

    int tx1 = floor(floor(x - r_fringe) / TILE_SIZE);
    int tx2 = floor(floor(x + r_fringe) / TILE_SIZE);
    int ty1 = floor(floor(y - r_fringe) / TILE_SIZE);
    int ty2 = floor(floor(y + r_fringe) / TILE_SIZE);
    int tx, ty;
    for (ty = ty1; ty <= ty2; ty++) {
      for (tx = tx1; tx <= tx2; tx++) {

        uint16_t * rgba_p = mypaint_tiled_surface_get_tile(self, tx, ty, FALSE);
        if (!rgba_p) {
          printf("Warning: Unable to get tile!\n");
          return TRUE;
        }

        // first, we calculate the mask (opacity for each pixel)
        static uint16_t mask[TILE_SIZE*TILE_SIZE+2*TILE_SIZE];

        render_dab_mask(mask,
                        x - tx*TILE_SIZE,
                        y - ty*TILE_SIZE,
                        radius,
                        hardness,
                        aspect_ratio, angle
                        );

        // second, we use the mask to stamp a dab for each activated blend mode

        if (normal) {
          if (color_a == 1.0) {
            draw_dab_pixels_BlendMode_Normal(mask, rgba_p,
                                             color_r_, color_g_, color_b_, normal*opaque*(1<<15));
          } else {
            // normal case for brushes that use smudging (eg. watercolor)
            draw_dab_pixels_BlendMode_Normal_and_Eraser(mask, rgba_p,
                                                        color_r_, color_g_, color_b_, color_a*(1<<15), normal*opaque*(1<<15));
          }
        }

        if (lock_alpha) {
          draw_dab_pixels_BlendMode_LockAlpha(mask, rgba_p,
                                              color_r_, color_g_, color_b_, lock_alpha*opaque*(1<<15));
        }
        if (colorize) {
          draw_dab_pixels_BlendMode_Color(mask, rgba_p,
                                          color_r_, color_g_, color_b_,
                                          colorize*opaque*(1<<15));
        }

      mypaint_tiled_surface_update_tile(self, tx, ty, rgba_p);
      }
    }


    {
      // expand the bounding box to include the region we just drawed
      int bb_x, bb_y, bb_w, bb_h;
      bb_x = floor (x - (radius+1));
      bb_y = floor (y - (radius+1));
      /* FIXME: think about it exactly */
      bb_w = ceil (2*(radius+1));
      bb_h = ceil (2*(radius+1));

      mypaint_tiled_surface_area_changed(self, bb_x, bb_y, bb_w, bb_h);
    }

    if(!recursing && self->surface_do_symmetry) {
      draw_dab_internal (self, self->surface_center_x + (self->surface_center_x - x), y,
                radius,
                color_r, color_g, color_b,
                opaque, hardness,
                color_a,
                aspect_ratio, -angle,
                lock_alpha,
                colorize,
                1);
    }

    return TRUE;
  }

// returns TRUE if the surface was modified
int draw_dab (MyPaintSurface *surface, float x, float y,
               float radius,
               float color_r, float color_g, float color_b,
               float opaque, float hardness,
               float color_a,
               float aspect_ratio, float angle,
               float lock_alpha,
               float colorize)
{
  MyPaintTiledSurface *self = (MyPaintTiledSurface *)surface;
  return draw_dab_internal(self, x, y, radius, color_r, color_g, color_b,
                           opaque, hardness, color_a, aspect_ratio, angle,
                           lock_alpha, colorize, 0);
}

void get_color (MyPaintSurface *surface, float x, float y,
                  float radius,
                  float * color_r, float * color_g, float * color_b, float * color_a
                  )
{
    MyPaintTiledSurface *self = (MyPaintTiledSurface *)surface;

    float r_fringe;

    if (radius < 1.0) radius = 1.0;
    const float hardness = 0.5;
    const float aspect_ratio = 1.0;
    const float angle = 0.0;

    float sum_weight, sum_r, sum_g, sum_b, sum_a;
    sum_weight = sum_r = sum_g = sum_b = sum_a = 0.0;

    // in case we return with an error
    *color_r = 0.0;
    *color_g = 1.0;
    *color_b = 0.0;

    // WARNING: some code duplication with draw_dab

    r_fringe = radius + 1;

    int tx1 = floor(floor(x - r_fringe) / TILE_SIZE);
    int tx2 = floor(floor(x + r_fringe) / TILE_SIZE);
    int ty1 = floor(floor(y - r_fringe) / TILE_SIZE);
    int ty2 = floor(floor(y + r_fringe) / TILE_SIZE);
    int tx, ty;
    for (ty = ty1; ty <= ty2; ty++) {
      for (tx = tx1; tx <= tx2; tx++) {
        uint16_t * rgba_p = mypaint_tiled_surface_get_tile(self, tx, ty, TRUE);
        if (!rgba_p) {
          printf("Warning: Unable to get tile!\n");
          return;
        }

        // first, we calculate the mask (opacity for each pixel)
        static uint16_t mask[TILE_SIZE*TILE_SIZE+2*TILE_SIZE];

        render_dab_mask(mask,
                        x - tx*TILE_SIZE,
                        y - ty*TILE_SIZE,
                        radius,
                        hardness,
                        aspect_ratio, angle
                        );

        get_color_pixels_accumulate (mask, rgba_p,
                                     &sum_weight, &sum_r, &sum_g, &sum_b, &sum_a);

      }
    }

    assert(sum_weight > 0.0);
    sum_a /= sum_weight;
    sum_r /= sum_weight;
    sum_g /= sum_weight;
    sum_b /= sum_weight;

    *color_a = sum_a;
    // now un-premultiply the alpha
    if (sum_a > 0.0) {
      *color_r = sum_r / sum_a;
      *color_g = sum_g / sum_a;
      *color_b = sum_b / sum_a;
    } else {
      // it is all transparent, so don't care about the colors
      // (let's make them ugly so bugs will be visible)
      *color_r = 0.0;
      *color_g = 1.0;
      *color_b = 0.0;
    }

    // fix rounding problems that do happen due to floating point math
    *color_r = CLAMP(*color_r, 0.0, 1.0);
    *color_g = CLAMP(*color_g, 0.0, 1.0);
    *color_b = CLAMP(*color_b, 0.0, 1.0);
    *color_a = CLAMP(*color_a, 0.0, 1.0);
}

void
mypaint_tiled_surface_init(MyPaintTiledSurface *self)
{
    self->parent.draw_dab = draw_dab;
    self->parent.get_color = get_color;

    self->surface_do_symmetry = FALSE;
    self->surface_center_x = 0.0;
}

void
mypaint_tiled_surface_destroy(MyPaintTiledSurface *self)
{

}
