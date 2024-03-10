/* brushlib - The MyPaint Brush Library
 * Copyright (C) 2007-2008 Martin Renold <martinxyz@gmx.ch>
 * Copyright (C) 2012-2016 by the MyPaint Development Team
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef MAPPING_HPP
#define MAPPING_HPP

#include <mypaint-mapping.h>

// user-defined mappings
// (the curves you can edit in the brush settings)
class MappingWrapper {

public:
  MappingWrapper(int inputs_) {
      c_mapping = mypaint_mapping_new(inputs_);
  }
  ~MappingWrapper() {
      mypaint_mapping_free(c_mapping);
  }

  void set_n (int input, int n)
  {
      mypaint_mapping_set_n(c_mapping, input, n);
  }

  void set_point (int input, int index, float x, float y)
  {
      mypaint_mapping_set_point(c_mapping, input, index, x, y);
  }

  bool is_constant()
  {
    return mypaint_mapping_is_constant(c_mapping);
  }

  float calculate (float * data)
  {
      return mypaint_mapping_calculate(c_mapping, data);
  }

  // used in python for the global pressure mapping
  float calculate_single_input (float input)
  {
      return mypaint_mapping_calculate_single_input(c_mapping, input);
  }
private:
  MyPaintMapping *c_mapping;
};

#endif //MAPPING_HPP

