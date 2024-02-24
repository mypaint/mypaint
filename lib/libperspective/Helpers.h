/*
    This file is part of libPerspective.
    Copyright (C) 2019  Grzegorz Wójcik

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
#pragma once

#include "Projection.h"

struct BBox {
    precission min_x, min_y, max_x, max_y;
};

inline BBox get_line_bounding_box( const std::vector<Complex> & segments) {
    precission min_x = segments[0].real();
    precission min_y = segments[0].imag();
    precission max_x = min_x;
    precission max_y = min_y;
    for (auto && pos : segments) {
        min_x = pos.real() < min_x ? pos.real() : min_x;
        min_y = pos.imag() < min_y ? pos.imag() : min_y;
        max_x = pos.real() > max_x ? pos.real() : max_x;
        max_y = pos.imag() > max_y ? pos.imag() : max_y;
    }
    return BBox {
        .min_x = min_x,
        .min_y = min_y,
        .max_x = max_x,
        .max_y = max_y,
    };
}

inline std::vector<Quaternion> create_circle(const Quaternion & center, const Quaternion & start, const Quaternion & normal, int side_count) {
    precission step_angle = 2 * M_PI / side_count;
    Quaternion rotation = createRotationQuatenion(normal, step_angle);

    Quaternion pos = start - center;
    std::vector<Quaternion> circle;
    circle.push_back(pos + center);
    Quaternion forward = Quaternion::FORWARD();
    for (int i = 0; i < side_count; i++) {
        pos = rotate(rotation, pos);
        Quaternion edge_pos = pos + center;
        if (forward.dot_3D(edge_pos) > 0) {
            circle.push_back(edge_pos);
        }
    }

    return circle;
}

/** Divide sides of polygon, used for Curvilinear perspective */
inline std::vector<Quaternion> teselate_poligon (const std::vector<Quaternion> & polygon, int teselation) {
    std::vector<Quaternion> result;
    Quaternion start = polygon.back();
    result.push_back(start);
    for ( auto && point : polygon) {
        Quaternion diff = point - start;
        Quaternion step = diff.scalar_mul(1.0 / teselation);
        for (int i=0; i< teselation; i++) {
            result.push_back(start + step.scalar_mul(i));
        }
        start = point;
    }
    return result;
}

// TODO check description
/**
 * Find intersection between plane and ray.
 * Ray is assumed to start in point (0, 0, 1)
 * Used equation: R * (-(S-O)·N)/(V·N))
 * where R - ray, S - start position, N - plane normal, O - offset
 */
inline Quaternion intersect_view_ray_and_plane(const Quaternion & plane_normal, const Quaternion & offset, const Quaternion & ray) {
    precission ray_length = plane_normal.dot_3D(offset) / ray.dot_3D(plane_normal);
    return ray.scalar_mul(ray_length);
}

/** Plane class represents set of all possible planes perpendicular to its normal vector. */
class Plane {
private:
    Quaternion normal;
public:
    Plane() : normal(0, 1, 0, 0){};
    explicit Plane(const Quaternion & normal) : normal(normal) {};

    /** returns vector perpendicular to plane */
    Quaternion get_normal() {
        return normal;
    }

    /** sets vector perpendicular to plane */
    void set_normal(const Quaternion & normal) {
        this->normal = normal;
    }
};


/** groups nodes in perspective graph */
class PerspectiveGroup {
};
