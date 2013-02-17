/* brushlib - The MyPaint Brush Library
 * Copyright (C) 2008 Martin Renold <martinxyz@gmx.ch>
 * Copyright (C) 2012 Jon Nordby <jononor@gmail.com>
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

#include <mypaint-rectangle.h>
#include <stdlib.h>
#include <string.h>

void *memdup(const void *src, size_t len)
{
        void *p = malloc(len);
        if (p)
            memcpy(p, src, len);
        return p;
}

MyPaintRectangle *
mypaint_rectangle_copy(MyPaintRectangle *self)
{
    return (MyPaintRectangle *)memdup(self, sizeof(MyPaintRectangle));
}

void
mypaint_rectangle_expand_to_include_point(MyPaintRectangle *r, int x, int y)
{
    if (r->width == 0) {
        r->width = 1; r->height = 1;
        r->x = x; r->y = y;
    } else {
        if (x < r->x) { r->width += r->x-x; r->x = x; } else
        if (x >= r->x+r->width) { r->width = x - r->x + 1; }

        if (y < r->y) { r->height += r->y-y; r->y = y; } else
        if (y >= r->y+r->height) { r->height = y - r->y + 1; }
    }
}
