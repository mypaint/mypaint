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
#include <stdlib.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mypaint-tiled-surface.h"
#include "helpers.h"
#include "brushmodes.h"
#include "operationqueue.h"

#define M_PI 3.14159265358979323846

#define TILE_SIZE 64
#define MAX_MIPMAP_LEVEL 4

void process_tile(MyPaintTiledSurface *self, int tx, int ty);

void mypaint_tiled_surface_begin_atomic(MyPaintTiledSurface *self)
{

}

void mypaint_tiled_surface_end_atomic(MyPaintTiledSurface *self)
{
    // Process tiles
    TileIndex *tiles;
    int tiles_n = operation_queue_get_dirty_tiles(self->operation_queue, &tiles);

    #pragma omp parallel for schedule(static) if(tiles_n > 3)
    for (int i = 0; i < tiles_n; i++) {
        process_tile(self, tiles[i].x, tiles[i].y);
    }

    operation_queue_clear_dirty_tiles(self->operation_queue);
}

void mypaint_tiled_surface_tile_request_start(MyPaintTiledSurface *self, MyPaintTiledSurfaceTileRequestData *request)
{
    assert(self->tile_request_start);
    self->tile_request_start(self, request);
}


void mypaint_tiled_surface_tile_request_end(MyPaintTiledSurface *self, MyPaintTiledSurfaceTileRequestData *request)
{
    assert(self->tile_request_end);
    self->tile_request_end(self, request);
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

void
mypaint_tiled_surface_tile_request_init(MyPaintTiledSurfaceTileRequestData *data,
                                        int tx, int ty, gboolean readonly)
{
    data->tx = tx;
    data->ty = ty;
    data->readonly = readonly;
    data->buffer = NULL;
    data->context = NULL;
}

inline float
calculate_rr(int xp, int yp, float x, float y, float aspect_ratio,
                      float sn, float cs, float one_over_radius2)
{
    // code duplication, see brush::count_dabs_to()
    float yy = (yp + 0.5 - y);
    float xx = (xp + 0.5 - x);
    float yyr=(yy*cs-xx*sn)*aspect_ratio;
    float xxr=yy*sn+xx*cs;
    float rr = (yyr*yyr + xxr*xxr) * one_over_radius2;
    // rr is in range 0.0..1.0*sqrt(2)
    return rr;
}

inline float
calculate_opa(float rr, float hardness,
              float segment1_offset, float segment1_slope,
              float segment2_offset, float segment2_slope) {

    float fac = rr <= hardness ? segment1_slope : segment2_slope;
    float opa = rr <= hardness ? segment1_offset : segment2_offset;
    opa += rr*fac;

#ifdef HEAVY_DEBUG
    assert(isfinite(opa));
    assert(opa >= 0.0 && opa <= 1.0);
#endif

    if (rr > 1.0) {
        opa = 0.0;
    }
    return opa;
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

    float r_fringe = radius + 1;
    int x0 = floor (x - r_fringe);
    int y0 = floor (y - r_fringe);
    int x1 = ceil (x + r_fringe);
    int y1 = ceil (y + r_fringe);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > TILE_SIZE-1) x1 = TILE_SIZE-1;
    if (y1 > TILE_SIZE-1) y1 = TILE_SIZE-1;
    float one_over_radius2 = 1.0/(radius*radius);

    // Pre-calculate rr and put it in the mask.
    // This an optimization that makes use of auto-vectorization
    // OPTIMIZE: if using floats for the brush engine, store these directly in the mask
    float rr_mask[TILE_SIZE*TILE_SIZE+2*TILE_SIZE];

    for (int yp = y0; yp <= y1; yp++) {
      for (int xp = x0; xp <= x1; xp++) {
        float rr = calculate_rr(xp, yp,
                                x, y, aspect_ratio,
                                sn, cs, one_over_radius2);
        rr_mask[(yp*TILE_SIZE)+xp] = rr;
      }
    }

    // we do run length encoding: if opacity is zero, the next
    // value in the mask is the number of pixels that can be skipped.
    uint16_t * mask_p = mask;
    int skip=0;

    skip += y0*TILE_SIZE;
    for (int yp = y0; yp <= y1; yp++) {
      skip += x0;

      int xp;
      for (xp = x0; xp <= x1; xp++) {
        float rr = rr_mask[(yp*TILE_SIZE)+xp];
        float opa = calculate_opa(rr, hardness,
                                  segment1_offset, segment1_slope,
                                  segment2_offset, segment2_slope);
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

void
process_op(MyPaintTiledSurface *self, uint16_t *rgba_p, uint16_t *mask,
           int tx, int ty, OperationDataDrawDab *op)
{

    // first, we calculate the mask (opacity for each pixel)
    render_dab_mask(mask,
                    op->x - tx*TILE_SIZE,
                    op->y - ty*TILE_SIZE,
                    op->radius,
                    op->hardness,
                    op->aspect_ratio, op->angle
                    );

    // second, we use the mask to stamp a dab for each activated blend mode

    if (op->normal) {
      if (op->color_a == 1.0) {
        draw_dab_pixels_BlendMode_Normal(mask, rgba_p,
                                         op->color_r, op->color_g, op->color_b, op->normal*op->opaque*(1<<15));
      } else {
        // normal case for brushes that use smudging (eg. watercolor)
        draw_dab_pixels_BlendMode_Normal_and_Eraser(mask, rgba_p,
                                                    op->color_r, op->color_g, op->color_b, op->color_a*(1<<15), op->normal*op->opaque*(1<<15));
      }
    }

    if (op->lock_alpha) {
      draw_dab_pixels_BlendMode_LockAlpha(mask, rgba_p,
                                          op->color_r, op->color_g, op->color_b, op->lock_alpha*op->opaque*(1<<15));
    }
    if (op->colorize) {
      draw_dab_pixels_BlendMode_Color(mask, rgba_p,
                                      op->color_r, op->color_g, op->color_b,
                                      op->colorize*op->opaque*(1<<15));
    }

    // FIXME: only emit on end_atomic
    {
      // expand the bounding box to include the region we just drawed
      int bb_x, bb_y, bb_w, bb_h;
      bb_x = floor (op->x - (op->radius+1));
      bb_y = floor (op->y - (op->radius+1));
      /* FIXME: think about it exactly */
      bb_w = ceil (2*(op->radius+1));
      bb_h = ceil (2*(op->radius+1));

      mypaint_tiled_surface_area_changed(self, bb_x, bb_y, bb_w, bb_h);
    }
}

void
process_tile(MyPaintTiledSurface *self, int tx, int ty)
{
    TileIndex tile_index = {tx, ty};
    OperationDataDrawDab *op = operation_queue_pop(self->operation_queue, tile_index);
    if (!op) {
        return;
    }

    MyPaintTiledSurfaceTileRequestData request_data;
    mypaint_tiled_surface_tile_request_init(&request_data, tx, ty, FALSE);

    mypaint_tiled_surface_tile_request_start(self, &request_data);
    uint16_t * rgba_p = request_data.buffer;
    if (!rgba_p) {
        printf("Warning: Unable to get tile!\n");
        return;
    }

    uint16_t mask[TILE_SIZE*TILE_SIZE+2*TILE_SIZE];

    while (op) {
        process_op(self, rgba_p, mask, tile_index.x, tile_index.y, op);
        free(op);
        op = operation_queue_pop(self->operation_queue, tile_index);
    }

    mypaint_tiled_surface_tile_request_end(self, &request_data);
}

// returns TRUE if the surface was modified
gboolean draw_dab_internal (MyPaintTiledSurface *self, float x, float y,
               float radius,
               float color_r, float color_g, float color_b,
               float opaque, float hardness,
               float color_a,
               float aspect_ratio, float angle,
               float lock_alpha,
               float colorize
               )

{
    OperationDataDrawDab op_struct;
    OperationDataDrawDab *op = &op_struct;

    op->x = x;
    op->y = y;
    op->radius = radius;
    op->aspect_ratio = aspect_ratio;
    op->angle = angle;
    op->opaque = CLAMP(opaque, 0.0, 1.0);
    op->hardness = CLAMP(hardness, 0.0, 1.0);
    op->lock_alpha = CLAMP(lock_alpha, 0.0, 1.0);
    op->colorize = CLAMP(colorize, 0.0, 1.0);
    if (op->radius < 0.1) return FALSE; // don't bother with dabs smaller than 0.1 pixel
    if (op->hardness == 0.0) return FALSE; // infintly small center point, fully transparent outside
    if (op->opaque == 0.0) return FALSE;

    color_r = CLAMP(color_r, 0.0, 1.0);
    color_g = CLAMP(color_g, 0.0, 1.0);
    color_b = CLAMP(color_b, 0.0, 1.0);
    color_a = CLAMP(color_a, 0.0, 1.0);

    op->color_r = color_r * (1<<15);
    op->color_g = color_g * (1<<15);
    op->color_b = color_b * (1<<15);
    op->color_a = color_a;

    // blending mode preparation
    op->normal = 1.0;

    op->normal *= 1.0-op->lock_alpha;
    op->normal *= 1.0-op->colorize;

    if (op->aspect_ratio<1.0) op->aspect_ratio=1.0;

    // Determine the tiles influenced by operation, and queue it for processing for each tile
    {
    float r_fringe = op->radius + 1;

    int tx1 = floor(floor(x - r_fringe) / TILE_SIZE);
    int tx2 = floor(floor(x + r_fringe) / TILE_SIZE);
    int ty1 = floor(floor(y - r_fringe) / TILE_SIZE);
    int ty2 = floor(floor(y + r_fringe) / TILE_SIZE);
    int tx, ty;

    for (ty = ty1; ty <= ty2; ty++) {
      for (tx = tx1; tx <= tx2; tx++) {
          const TileIndex tile_index = {tx, ty};
          OperationDataDrawDab *op_copy = (OperationDataDrawDab *)malloc(sizeof(OperationDataDrawDab));
          *op_copy = *op;
          operation_queue_add(self->operation_queue, tile_index, op_copy);
      }
    }

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

  gboolean surface_modified = FALSE;

  // Normal pass
  if (draw_dab_internal(self, x, y, radius, color_r, color_g, color_b,
                        opaque, hardness, color_a, aspect_ratio, angle,
                        lock_alpha, colorize)) {
      surface_modified = TRUE;
  }

  // Symmetry pass
  if(self->surface_do_symmetry) {
    const int symm_x = self->surface_center_x + (self->surface_center_x - x);

    if (draw_dab_internal(self, symm_x, y, radius, color_r, color_g, color_b,
                           opaque, hardness, color_a, aspect_ratio, -angle,
                           lock_alpha, colorize)) {
        surface_modified = TRUE;
    }

  }

  return surface_modified;
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
    int tiles_n = (tx2 - tx1) * (ty2 - ty1);

    #pragma omp parallel for schedule(static) if(tiles_n > 3)
    for (int ty = ty1; ty <= ty2; ty++) {
      for (int tx = tx1; tx <= tx2; tx++) {

        // Flush queued draw_dab operations
        process_tile(self, tx, ty);

        MyPaintTiledSurfaceTileRequestData request_data;
        mypaint_tiled_surface_tile_request_init(&request_data, tx, ty, TRUE);

        mypaint_tiled_surface_tile_request_start(self, &request_data);
        uint16_t * rgba_p = request_data.buffer;
        if (!rgba_p) {
          printf("Warning: Unable to get tile!\n");
          break;
        }

        // first, we calculate the mask (opacity for each pixel)
        uint16_t mask[TILE_SIZE*TILE_SIZE+2*TILE_SIZE];

        render_dab_mask(mask,
                        x - tx*TILE_SIZE,
                        y - ty*TILE_SIZE,
                        radius,
                        hardness,
                        aspect_ratio, angle
                        );

        // TODO: try atomic operations instead
        #pragma omp critical
        {
        get_color_pixels_accumulate (mask, rgba_p,
                                     &sum_weight, &sum_r, &sum_g, &sum_b, &sum_a);
        }

        mypaint_tiled_surface_tile_request_end(self, &request_data);
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
mypaint_tiled_surface_init(MyPaintTiledSurface *self,
                           MyPaintTiledSurfaceTileRequestStartFunction tile_request_start,
                           MyPaintTiledSurfaceTileRequestEndFunction tile_request_end)
{
    self->parent.draw_dab = draw_dab;
    self->parent.get_color = get_color;
    self->tile_request_end = tile_request_end;
    self->tile_request_start = tile_request_start;

    self->area_changed = NULL;
    self->threadsafe_tile_requests = FALSE;

    self->surface_do_symmetry = FALSE;
    self->surface_center_x = 0.0;
    self->operation_queue = operation_queue_new();
}

void
mypaint_tiled_surface_destroy(MyPaintTiledSurface *self)
{
    operation_queue_free(self->operation_queue);
}
