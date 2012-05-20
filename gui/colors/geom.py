# This file is part of MyPaint.
# Copyright (C) 2011-2012 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


""" Polygon and convex polygon computational geometry routines.
"""

from math import sqrt


def convex_hull(points):
    """Returns the convex hull of a set of points, in clockwise order.

      >>> convex_hull([(1,1), (1,-1), (0,0), (-1,-1), (-1,1)])
      [(-1, -1), (1, -1), (1, 1), (-1, 1)]

    Uses the Graham scan algorithm for finding the ordered set of points.
    Ref: http://en.wikipedia.org/wiki/Graham_scan
    Ref: http://cgm.cs.mcgill.ca/~beezer/cs507/3coins.html

    """

    # Uniquify
    points = dict.fromkeys(points).keys()

    # Extract the point whose Y-coordinate is lowest. Tiebreak using the lowest
    # X-coordinate.
    p0 = points[0]
    for p in points[1:]:
        if p[1] < p0[1] or (p[1] == p0[1] and p[0] < p0[0]):
            p0 = p
    points.remove(p0)

    # Sort other points clockwise in increasing order of the angle the vector
    # p0->p makes with the X axis. Or just the cosine, which suffices since
    # p0 has the lowest Y value and the angle is therefore in (0, pi).
    def p0cos(p):
        return (p0[0]-p[0]) / sqrt(((p0[0]-p[0])**2)+((p0[1]-p[1])**2))
    points = [(p0cos(p), p) for p in points]
    points.sort()
    points = [tup[1] for tup in points]
    points.insert(0, p0)

    # Build the hull as a stack, continually removing the middle element
    # of the last three points while those three points make a left turn
    # rather than a right turn.
    hull = points[0:2]

    def det(p, q, r):
        """Determinant of the vector pq:qr
        
        If pq:qr is a clockwise turn, result is negative. If the points
        are collinear, return zero.

        """
        sum1 = q[0]*r[1] + p[0]*q[1] + r[0]*p[1]
        sum2 = q[0]*p[1] + r[0]*q[1] + p[0]*r[1]
        return sum1 - sum2

    for p in points[2:]:
        hull.append(p)
        while len(hull) > 2 and det(*hull[-3:]) <= 0:
            del hull[-2]
    return hull


def pairwise(seq):
    """Pairwise sequence iterator.
    
      >>> list(pairwise("spam"))
      [('s', 'p'), ('p', 'a'), ('a', 'm'), ('m', 's')]

    Returns {seq[i],seq[i+1], ..., seq[n],seq[0]} for seq[0...n].

    """
    n = 0
    first_item = None
    prev_item = None
    for item in seq:
        if n == 0:
            first_item = item
        else:
            yield prev_item, item
        prev_item = item
        n += 1
    if n > 1:
        yield item, first_item


def poly_area(poly):
    """Calculates the area of a (non-self-intersecting) polygon.

      >>> poly_area([(-1, -1), (1, -1), (1, 1), (-1, 1)])
      4.0

    """
    area = 0.0
    for pa, pb in pairwise(poly):
        area += pa[0]*pb[1] - pb[0]*pa[1]
    area /= 2.0
    return area


def poly_centroid(poly):
    """Calculates the centroid of a (non-self-intersecting) polygon.

      >>> poly_centroid([(-1, -1), (1, -1), (1, 1), (-1, 1)])
      (0.0, 0.0)
      >>> poly_centroid([(0, 1), (0, 4), (0, 3)])
      (0.0, 2.5)

    """
    cx, cy = 0.0, 0.0
    area = 0.0
    for pa, pb in pairwise(poly):
        n = (pa[0]*pb[1] - pb[0]*pa[1])
        cx += (pa[0]+pb[0]) * n
        cy += (pa[1]+pb[1]) * n
        area += pa[0]*pb[1] - pb[0]*pa[1]
    if area > 0.0:
        area /= 2.0
        cx /= 6.0*area
        cy /= 6.0*area
        return cx, cy
    else:  # Line
        xs = [x for x,y in poly]
        ys = [y for x,y in poly]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        return cx, cy


def point_in_convex_poly(point, poly):
    """True if a point is inside a convex polygon.

      >>> square = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
      >>> point_in_convex_poly((0,0), square)
      True
      >>> point_in_convex_poly((0,1), square)
      True
      >>> point_in_convex_poly((0,1.1), square)
      False

    A point exactly on the boundary is considered to lie inside the
    polygon. `poly` can be described in either direction, but the points
    must be ordered. Ref: http://paulbourke.net/geometry/insidepoly/

    """
    x, y = point
    seen_left = seen_right = False
    for p0, p1 in pairwise(poly):
        x0, y0 = p0
        x1, y1 = p1
        det = ((y-y0)*(x1-x0)) - ((x-x0)*(y1-y0))
        if det < 0: # point lies to right of segment
            if seen_left:
                return False
            seen_right = True
        elif det > 0: # point lies to left of segment
            if seen_right:
                return False
            seen_left = True
        else: # point is on the same line as the segment
            pass
    return True


def nearest_point_in_segment(seg_start, seg_end, point):
    """Intersection of a segment & the line perpendicular to it through a point

    The points `seg_start` and `seg_end` bound the line segment. The return
    value is the point where this segment intersects the line perpendicular to
    it passing through `point`.

      >>> nearest_point_in_segment((0,0), (4,0), (2,2))
      (2.0, 0.0)
      >>> nearest_point_in_segment((1,1), (3,3), (2,1))
      (1.5, 1.5)

    If the points `p1` and `p2` are coincident, or the intersection would lie
    outside the segment, `None` is returned.

      >>> nearest_point_in_segment((1,1), (3,3), (12,-1)) is None # not in seg
      True
      >>> nearest_point_in_segment((1,1), (1,1), (2,2)) is None # coincident
      True

    Ref: http://paulbourke.net/geometry/pointline/

    """
    x1, y1 = [float(n) for n in seg_start]
    x2, y2 = [float(n) for n in seg_end]
    x3, y3 = [float(n) for n in point]
    denom = (x2-x1)**2 + (y2-y1)**2
    if denom == 0:
        return None  # seg_start and seg_end are coincident
    u = ((x3 - x1)*(x2 - x1) + (y3 - y1)*(y2 - y1)) / denom
    if u <= 0 or u >= 1:
        return None  # intersection is not within the line segment
    x = x1 + u*(x2-x1)
    y = y1 + u*(y2-y1)
    return x, y



if __name__ == '__main__':
    import doctest
    doctest.testmod()

