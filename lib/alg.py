# This file is part of MyPaint.
# Copyright (C) 2011-2014 by Andrew Chadwick <a.t.chadwick@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Miscellaneous little algorithms"""

## Imports
from __future__ import division, print_function

from math import sqrt


class LineType:
    LINE = 0  # Infinite line
    DIRECTIONAL = 1  # Infinite in one direction
    SEGMENT = 2  # Fixed segment


## Polygon and convex polygon computational geometry routines.

def convex_hull(points):
    """Returns the convex hull of a set of points, in clockwise order.

      >>> convex_hull([(1,1), (1,-1), (0,0), (-1,-1), (-1,1)])
      [(-1, -1), (1, -1), (1, 1), (-1, 1)]

    Uses the Graham scan algorithm for finding the ordered set of points.
    Ref: http://en.wikipedia.org/wiki/Graham_scan
    Ref: http://cgm.cs.mcgill.ca/~beezer/cs507/3coins.html

    """

    # Uniquify
    points = list(dict.fromkeys(points).keys())

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
        return ((p0[0] - p[0]) / sqrt((p0[0] - p[0])**2 + (p0[1] - p[1])**2),
                p)

    points = sorted(points, key=p0cos)
    points.insert(0, p0)

    # Build the hull as a stack, continually removing the middle element
    # of the last three points while those three points make a left turn
    # rather than a right turn.
    hull = points[0:2]

    for p in points[2:]:
        hull.append(p)
        while len(hull) > 2 and det(*hull[-3:]) <= 0:
            del hull[-2]
    return hull


def det(p, q, r):
    """Determinant of the vector pq:qr

    If pq:qr is a clockwise turn, result is negative. If the points
    are collinear, return zero.

    """
    sum1 = q[0]*r[1] + p[0]*q[1] + r[0]*p[1]
    sum2 = q[0]*p[1] + r[0]*q[1] + p[0]*r[1]
    return sum1 - sum2


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
        xs = [x for x, y in poly]
        ys = [y for x, y in poly]
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
        if det < 0:  # point lies to right of segment
            if seen_left:
                return False
            seen_right = True
        elif det > 0:  # point lies to left of segment
            if seen_right:
                return False
            seen_left = True
        else:  # point is on the same line as the segment
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
      >>> nearest_point_in_segment((0,0), (3,0), (0,1))

    If the points `p1` and `p2` are coincident, or the intersection would lie
    outside the segment, `None` is returned.

      >>> nearest_point_in_segment((1,1), (3,3), (12,-1)) is None # not in seg
      True
      >>> nearest_point_in_segment((1,1), (1,1), (2,2)) is None # coincident
      True

    Ref: http://paulbourke.net/geometry/pointline/

    """
    return _nearest_point(seg_start, seg_end, point)


def nearest_point_on_segment(seg_start, seg_end, point):
    """Get the point on a segment closest to the given point

    The points `seg_start` and `seg_end` bound the line segment. The return
    value is either the point where the segment intersects the line
    perpendicular to it passing through `point`, or whichever end
    of the segment is closer to the point.

      >>> nearest_point_on_segment((0, 0), (0, 4), (0, 2))
      (0.0, 2.0)
      >>> nearest_point_on_segment((0, 0), (0, 4), (2, 3))
      (0.0, 3.0)
      >>> nearest_point_on_segment((0, 0), (4, 0), (-2, 3))
      (0.0, 0.0)
      >>> nearest_point_on_segment((0, 0), (4, 0), (-2, -5))
      (0.0, 0.0)
      >>> nearest_point_on_segment((0, 0), (4, 0), (6, 5))
      (4.0, 0.0)
    """
    return _nearest_point(seg_start, seg_end, point, perpendicular=False)


def nearest_point_in_poly(poly, point):
    """Return the point in a given convex polygon closest to the given point.

    >>> poly = [(-3, 1), (2, 3), (4, 1), (2, -4)]
    >>> nearest_point_in_poly(poly, (-4, -2))
    (-2.0, 0.0)
    >>> nearest_point_in_poly(poly, (-5, 1))
    (-3.0, 1.0)
    >>> nearest_point_in_poly(poly, (0, 2))
    (0, 2)
    >>> nearest_point_in_poly(poly, (4, 3))
    (3.0, 2.0)
    >>> nearest_point_in_poly(poly, (6, 2))
    (4.0, 1.0)
    """

    if point_in_convex_poly(point, poly):
        return point

    closest = None
    smallest_dist_sqr = float("inf")
    x0, y0 = point
    for p0, p1 in pairwise(poly):
        candidate = nearest_point_on_segment(p0, p1, point)
        if candidate:
            x1, y1 = candidate
            dist_sqr = (x1 - x0) ** 2 + (y1 - y0) ** 2
            if dist_sqr < smallest_dist_sqr:
                smallest_dist_sqr = dist_sqr
                closest = candidate
    return closest


def nearest_point_on_line(p1, p2, point, unidirectional=False):
    """For a line l and a point p, return the point on l closest to p

    The line is defined by two pairs of coordinates, and is either considered
    infinite in both directions, or in one direction (starting at the first
    coordinates) if `unidirectional` is set to True.

    If `unidirectional` is False, and the line is valid, the closest point
    will always be returned. If `unidirectional` is True and the closest point
    on the line is not perpendicular to `point`, None is returned.

    Line ends are inclusive (only relevant for directional lines), meaning that
    for e.g. the line (0,0), (1, 1) and point (2, 0), the closest point will be
    (1, 1) and not None.
    """
    return _nearest_point(
        p1, p2, point, inclusive=True, line_type=int(unidirectional))


def _nearest_point(
        seg_start, seg_end, point,
        perpendicular=True, inclusive=False, line_type=LineType.SEGMENT):
    """Generic impl, supporting non-perpendicular shortest distance

      >>> _nearest_point((0, 0), (3, 0), (0, 1), inclusive=True)
      (0.0, 0.0)
      >>> _nearest_point((0, 0), (3, 0), (0, 1), perpendicular=False)
      (0.0, 0.0)
      >>> _nearest_point((0, 0), (3, 0), (-1, 1), perpendicular=False)
      (0.0, 0.0)
      >>> _nearest_point((3, 0), (0, 0), (-1, 1), perpendicular=False)
      (0.0, 0.0)
      >>> _nearest_point((0, 0), (3, 0), (4, 1), perpendicular=False)
      (3.0, 0.0)
      >>> _nearest_point((3, 0), (0, 0), (4, 1), perpendicular=False)
      (3.0, 0.0)
      >>> _nearest_point((-1, 1), (1, -1), (-3, 1), perpendicular=False)
      (-1.0, 1.0)
      >>> _nearest_point((-1, 1), (1, -1), (-1, 3), perpendicular=False)
      (-1.0, 1.0)
      >>> _nearest_point((1, -1), (-1, 1), (-3, 1), perpendicular=False)
      (-1.0, 1.0)
      >>> _nearest_point((1, -1), (-1, 1), (-1, 3), perpendicular=False)
      (-1.0, 1.0)
      >>> _nearest_point((0, 0), (-2, 3), (-1, -5))
      >>> _nearest_point((0, 0), (-2, 3), (-1, -5), perpendicular=False)
      (0.0, 0.0)
      >>> _nearest_point((0, 0), (-2, 3), (-1, -5), line_type=LineType.LINE)
      (2.0, -3.0)
      >>> _nearest_point((0, 0), (-2, 3), (-1, -5),
      ...                line_type=LineType.DIRECTIONAL)
      >>> _nearest_point((-2, 3), (0, 0), (-1, -5),
      ...                line_type=LineType.DIRECTIONAL)
      (2.0, -3.0)
    """
    x1, y1 = [float(n) for n in seg_start]
    x2, y2 = [float(n) for n in seg_end]
    x3, y3 = [float(n) for n in point]
    denominator = (x2-x1)**2 + (y2-y1)**2
    if denominator == 0:
        return None  # seg_start and seg_end are coincident
    u = ((x3 - x1)*(x2 - x1) + (y3 - y1)*(y2 - y1)) / denominator
    outside = not inclusive and not (0 < u < 1)
    outside = outside or (inclusive and not (0 <= u <= 1))

    if outside and perpendicular:
        if (line_type == LineType.SEGMENT
                or line_type == LineType.DIRECTIONAL and
                (inclusive and u < 0 or not inclusive and u <= 0)):
            return None
    elif outside:
        return (x1, y1) if u <= 0 else (x2, y2)

    return x1 + u * (x2 - x1), y1 + u * (y2 - y1)


def intersection_of_vector_and_poly(
        poly, p1, p2, line_type=LineType.LINE):
    """Intersection of a vector and a convex polygon

    Returns two coordinate pairs indicating the section of the line that
    lies within the polygon, or None if there either is no intersection,
    or the line is only tangential to a single point.

    :param poly: An ordered sequence of coordinates defining a convex polygon
    :param p1: The first point of the vector/line
    :param p2: The second point of the vector/line
    :param line_type: The type of line/segment to intersect with the polygon.

      >>> isect = intersection_of_vector_and_poly
      >>> poly = [(-5, -5), (-7, 1), (-1, 3), (1, -2)]
      >>> p1, p2 = (-4, -2), (-4, 1)
      >>> isect(poly, p1, p2)
      [(-4.0, 2.0), (-4.0, -4.5)]
      >>> isect(poly, p1, p2, line_type=LineType.SEGMENT)
      [(-4, -2), (-4, 1)]
      >>> isect(poly, p1, p2, line_type=LineType.DIRECTIONAL)
      [(-4, -2), (-4.0, 2.0)]
      >>> isect(poly, p2, p1, line_type=LineType.DIRECTIONAL)
      [(-4, 1), (-4.0, -4.5)]

    Tangential to a single point
      >>> q1, q2 = (-7, -2), (-3, -8)
      >>> isect(poly, q1, q2)

    Coinciding with edge, infinite
      >>> w1, w2 = (), ()
      >>> isect(poly, (-7, 1), (-4, 2))
      [(-7.0, 1.0), (-1.0, 3.0)]

    Coinciding with edge, directional
      >>> isect(poly, (-7, 1), (-4, 2), LineType.DIRECTIONAL)
      [(-7.0, 1.0), (-1.0, 3.0)]

    Coinciding with edge, directional (reversed direction)
      >>> isect(poly, (-4, 2), (-7, 1), LineType.DIRECTIONAL)
      [(-4, 2), (-7.0, 1.0)]
    """

    result = []
    prev = None
    # Calculate all of the intersections, which should be at most 2
    # Since a line can intersect with two segments in a corner, such instances
    # are not added.
    for q1, q2 in pairwise(poly):
        inter = intersection_of_vectors(
            p1, p2, q1, q2, a_type=line_type, b_type=LineType.SEGMENT)
        if inter and inter != prev:
            result.append(inter)
            prev = inter
    # No edge intersections does not imply that there is _no_ intersection,
    # as it can be a case of a segment lying entirely within the polygon.
    if not result and line_type == LineType.SEGMENT:
        if point_in_convex_poly(p1, poly) and point_in_convex_poly(p2, poly):
            result = [p1, p2]
    # A single intersection either means that the line only intersects
    # a single corner or that there is an endpoint that lies inside
    # the polygon. If the latter is the case, that point is included.
    elif len(result) == 1:
        if line_type == LineType.DIRECTIONAL:
            if point_in_convex_poly(p1, poly) and p1 != prev:
                result.insert(0, p1)
        elif line_type == LineType.SEGMENT:
            if point_in_convex_poly(p1, poly) and p1 != prev:
                result.insert(0, p1)
            elif point_in_convex_poly(p2, poly) and p2 != prev:
                result.append(p2)
    if len(result) != 2:
        return None
    else:
        return result


def _intersects(line_type, k):
    """Helper function used to distinguish handling of segment/line types"""
    if line_type == LineType.LINE:
        return True  # Unless parallel/coinciding, lines always intersect
    elif line_type == LineType.DIRECTIONAL:
        return 0 <= k
    else:
        return 0 <= k <= 1


def intersection_of_vectors(
        a1, a2, b1, b2,
        a_type=LineType.LINE,
        b_type=LineType.LINE,
):
    """ Intersection of two vectors, interpreted as segments or lines

    The two vectors are defined by four coordinate pairs, and each can be
    either interpreted as a finite segment, or an infinite line which is either
    unidirectional or bidirectional. By default, both vectors are interpreted
    as defining bidirectional infinite lines.

    If there is an intersection, its coordinate is returned, otherwise
    (if the vectors coincide, are parallel, or never cross) None is returned.

    :param a1: The first point of the first vector
    :param a2: The second point of the first vector
    :param b1: The first point of the second vector
    :param b2: The second point of the second vector
    :param a_type: The type of the first vector
    :param b_type: The type of the second vector
    :return: an (x, y) tuple or None

      Two segments that do not intersect, but whose bidir infinite lines do,
      and, where if unidirectional, the second line intersects the first
      segment if the second stretches in one direction, but not the other.
      >>> isect = intersection_of_vectors
      >>> SEGMENT = LineType.SEGMENT
      >>> DIR = LineType.DIRECTIONAL
      >>> a1, a2 = (-1, 1), (-4, 4)
      >>> b1, b2 = (0, 6), (-2, 4)
      >>> intersection_of_vectors(a1, a2, b1, b2, SEGMENT, SEGMENT)
      >>> isect(a1, a2, b1, b2)
      (-3.0, 3.0)
      >>> isect(a1, a2, b1, b2, a_type=DIR)
      (-3.0, 3.0)
      >>> isect(a2, a1, b1, b2, a_type=DIR)
      (-3.0, 3.0)
      >>> isect(a1, a2, b1, b2, b_type=DIR)
      (-3.0, 3.0)
      >>> isect(a1, a2, b2, b1, b_type=DIR)

      Two segments that intersect - should always intersect regardless of type.
      >>> c1, c2 = (0, -4), (6, 0)
      >>> d1, d2 = (1, 0), (5, -4)
      >>> i = (3.0, -2.0)
      >>> r = isect(c1, c2, d1, d2); None if r == i else r
      >>> r = isect(c1, c2, d1, d2, SEGMENT, SEGMENT); None if r == i else r
      >>> r = isect(c1, c2, d1, d2, SEGMENT, DIR); None if r == i else r
      >>> r = isect(c1, c2, d1, d2, DIR, SEGMENT); None if r == i else r
      >>> r = isect(c1, c2, d1, d2, DIR, DIR); None if r == i else r

      Two segments that do not intersect, whose infinite lines do, but where
      the single-direction infinite lines/segments only intersect if both
      segments have a specific direction.
      >>> e1, e2 = (-6, 0), (-4, -2)
      >>> f1, f2 = (0, 0), (-2, -2)
      >>> i = (-3.0, -3.0)
      >>> isect(e1, e2, f1, f2, a_type=SEGMENT)
      >>> isect(e1, e2, f1, f2, b_type=SEGMENT)
      >>> isect(e1, e2, f1, f2, SEGMENT, SEGMENT)
      >>> r = isect(e1, e2, f1, f2); None if r == i else r
      >>> r = isect(e1, e2, f1, f2, a_type=DIR); None if r == i else r
      >>> r = isect(e1, e2, f1, f2, b_type=DIR); None if r == i else r
      >>> r = isect(e1, e2, f1, f2, DIR, DIR); None if r == i else r
      >>> isect(e2, e1, f1, f2, a_type=DIR)
      >>> isect(e2, e1, f2, f1, a_type=DIR)
      >>> isect(e1, e2, f2, f1, b_type=DIR)
      >>> isect(e2, e1, f2, f1, b_type=DIR)

    """
    (x0, y0), (x1, y1) = a1, a2
    (x2, y2), (x3, y3) = b1, b2
    d = (y3 - y2) * (x1 - x0) - (x3 - x2) * (y1 - y0)
    if d == 0:
        return None
    u1 = ((x3 - x2) * (y0 - y2) - (y3 - y2) * (x0 - x2)) / d
    if not (
            _intersects(a_type, u1) and
            _intersects(b_type,
                        ((x1 - x0) * (y0 - y2) - (y1 - y0) * (x0 - x2)) / d)):
        return None
    return u1 * (x1 - x0) + x0, u1 * (y1 - y0) + y0


def intersection_of_segments(p1, p2, p3, p4):
    """Intersection of two segments

    :param tuple p1: point on segment A, ``(x, y)``
    :param tuple p2: point on segment A, ``(x, y)``
    :param tuple p3: point on segment B, ``(x, y)``
    :param tuple p4: point on segment B, ``(x, y)``
    :returns: The point of intersection of the two segments
    :rtype: tuple or None

    If the two segments cross, the intersection point is returned:

      >>> intersection_of_segments((0,1), (1,0), (0,0), (2,2))
      (0.5, 0.5)
      >>> intersection_of_segments((0,1), (1,0), (-1,-3), (1,3))
      (0.25, 0.75)

    The return value is ``None`` if the two segments do not intersect.

      >>> intersection_of_segments((0,1), (1,0), (0,2), (1,1)) is None
      True

    Ref: http://paulbourke.net/geometry/pointlineplane/
    Ref: https://en.wikipedia.org/wiki/Line-line_intersection

    """
    # Unpack and validate args
    x1, y1 = [float(c) for c in p1]
    x2, y2 = [float(c) for c in p2]
    x3, y3 = [float(c) for c in p3]
    x4, y4 = [float(c) for c in p4]

    # Something close enough to zero
    epsilon = 0.0001

    # We're solving the twin parameterized equations for segments:
    #   pa = p1 + ua (p2-p1)
    #   pb = p3 + ub (p4-p3)
    # where ua and ub are in [0, 1].
    # Expanding, and setting pa = pb, yields:
    #   x1 + ua (x2 - x1)  =  x3 + ub (x4 - x3)
    #   y1 + ua (y2 - y1)  =  y3 + ub (y4 - y3)
    # Solving gives equations for the intersection
    # ua=numera/denom and ub=numerb/denom, where:
    numera = (x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)
    numerb = (x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)

    # Zero(ish) in the denominator indicates either coincident lines
    # or parallel (and therefore nonitersecting) lines.
    if abs(denom) < epsilon:
        if abs(numera) < epsilon and abs(numerb) < epsilon:  # coincident
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            return (x, y)
        else:
            return None   # parallel

    # The intersection is defined in terms of the parameters ua and ub.
    # If these are outside their range, the intersection point lies
    # along the segments' lines, but not within the segment.
    ua = numera / denom
    ub = numerb / denom
    if not ((0 <= ua <= 1) and (0 <= ub <= 1)):
        return None

    # Within segments, so just expand to an actual point.
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return (x, y)


## Iterations


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


## Testing


if __name__ == '__main__':
    import doctest
    doctest.testmod()
