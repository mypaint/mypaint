/* This file is part of MyPaint.
 * Copyright (C) 2018 by the MyPaint Development Team.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef MORPHOLOGY_HPP
#define MORPHOLOGY_HPP

#include "fill_common.hpp"
#include "morphology_swig.hpp"

/*
  Chords make up the structuring elements used to perform morphological
  transformations (erosion/dilation etc.).

  Since the morph functions are using the Urbach-Wilkinson (UW) algorithm,
  instead of storing the actual chord lengths, an index to the length
  is stored, avoiding redundant lookups/tests for chords of the same
  lengths..
*/
struct chord {
    chord() : x_offset(0), length_index(0) {}
    chord(int x, int len_i) : x_offset(x), length_index(len_i){}
    int x_offset;
    int length_index;
};

// Comparison operation type used to template dilation/erosion
typedef chan_t op(chan_t, chan_t);

/*
  Initiates and stores data shared between tile morph operations
  and allocates space reused between morphs, specifically:

  Circular structuring element data - radius/height/chords

  Lookup table (height x N x unique_chord_lengths) for linear(-ish) morph

  Input array (N + radius*2)^2 storing the pixels necessary to perform morph
  for the given radius - rotated/updated whenever possible.

  Output array to store morphed alpha values (consider removing/replacing).
*/

class Morpher
{
  public:
    explicit Morpher(int radius);
    ~Morpher();
    template <chan_t init, chan_t lim, op cmp>
    void morph(bool can_update, PixelBuffer<chan_t>& dst);
    template <chan_t lim>
    bool can_skip(PixelBuffer<chan_t> buf);
    void initiate(bool can_update, GridVector input);
    bool input_fully_opaque();
    bool input_fully_transparent();

  private:
    void rotate_lut();
    template <op cmp>
    void populate_row(int, int);

    int radius; // structuring element radius
    int height; // structuring element height
    std::vector<chord> se_chords; // structuring element chords
    std::vector<int> se_lengths; // structuring element chord lengths
    chan_t*** lookup_table; // lookup table for UW algorithm (y-offset, x, type)
    chan_t** input; // input 2d array populated by 3x3 input tile grid
};

#endif //MORPHOLOGY_HPP
