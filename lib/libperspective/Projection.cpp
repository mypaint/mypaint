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
#include "Projection.h"

namespace {
class PerspectiveLineSimple : public PerspectiveLine {
private:
    Complex start_position;
    Complex direction;
    static constexpr precission dot_product(const Complex & a, const Complex & b) {
        return a.real() * b.real() + a.imag() * b.imag();
    }
    static constexpr Complex rotate_90(const Complex & a) {
        return Complex(-a.imag(), a.real());
    }
public:
    PerspectiveLineSimple(RectilinearProjection * projection, const VanishingPoint & vp, const Complex & start_position) {
        this->start_position = start_position;
        this->direction = projection->get_direction_2d(vp, start_position);
    }

    virtual precission get_distance(const Complex & position) override {
        Complex relative_pos = position - this->start_position;

        // project relative_pos on axis orthogonal to direction
        // and get length of projection
        return std::abs(dot_product(relative_pos, rotate_90(this->direction)));
    }

    virtual std::vector<Complex> get_line_points(const Complex & position) override {
        Complex relative_pos = position - this->start_position;
        precission line_length = dot_product(relative_pos, this->direction);

        std::vector<Complex> result;
        result.reserve(2);
        result.push_back(this->start_position);
        result.push_back(this->start_position + this->direction * line_length);
        return result;
    }
};
class PerspectiveLineCurvilinear : public PerspectiveLine {
private:
    Quaternion start_dir;
    Quaternion plane_normal;
    CurvilinearPerspective * projection;
public:
    PerspectiveLineCurvilinear(CurvilinearPerspective * projection, const VanishingPoint & vp, const Complex & start_position) {
        this->projection = projection;
        this->start_dir = projection->calc_direction(start_position);
        this->plane_normal = normalize(cross(start_dir, vp.get_direction()));
    }

    virtual precission get_distance(const Complex & position) override {
        Quaternion new_dir = projection->calc_direction(position);
        precission distance_3d = plane_normal.dot_3D(new_dir);
        Quaternion pos_3d_on_plane = new_dir - plane_normal.scalar_mul(distance_3d);
        Complex pos_2d_on_plane = projection->calc_pos_from_dir(pos_3d_on_plane);
        precission distance = std::hypot(
            pos_2d_on_plane.real() - position.real(),
            pos_2d_on_plane.imag() - position.imag()
        );
        return distance;
    }

    virtual std::vector<Complex> get_line_points(const Complex & position) override {
        Quaternion pos_3d = projection->calc_direction(position);
        precission pos_plane_dist = plane_normal.dot_3D(pos_3d);
        Quaternion end_dir = pos_3d - plane_normal.scalar_mul(pos_plane_dist);
        Quaternion end_test_dir = cross(plane_normal, end_dir);

        Quaternion begin_test_dir = cross(plane_normal, start_dir);

        precission step_angle = 2 * M_PI / 100;
        Quaternion rotation = createRotationQuatenion(plane_normal, step_angle);
        if (begin_test_dir.dot_3D(end_dir) < 0) {
            rotation = rotation.conjugate();
            end_test_dir = end_test_dir.conjugate();
        }

        std::vector<Quaternion> line_points;
        Quaternion pos = start_dir;
        line_points.push_back(pos);
        for (int i = 0; i < 200; i++) {
            pos = rotate(rotation, pos);
            if (end_test_dir.dot_3D(pos) > 0) {
                break;
            }
            line_points.push_back(pos);
        }
        line_points.push_back(end_dir);
        return projection->project_on_canvas(line_points);
    }
};
class HorizonLineRectilinear : public HorizonLineBase {
private:
    BaseProjection * projection;
    Complex horizon_anchor_pos;
    Complex horizon_dir_2d;
    bool is_inf;
public:
    HorizonLineRectilinear(BaseProjection * projection, const Quaternion & up) {
        precission projected_len = std::hypot(up.x, up.y);
        this->projection = projection;
        if (projected_len == 0 || up.z == 0) {
            // up is forward direction, no visible horizon
            this->is_inf = true;
            return;
        }
        this->is_inf = false;

        Complex projected_up_normalized = Complex(up.x, up.y)/projected_len;
        precission zz = up.z * up.z;
        precission horizon_distance = zz/projected_len;
        this->horizon_anchor_pos = -projected_up_normalized * horizon_distance * (1/up.z);
        Complex horizon_dir = Complex(up.y, -up.x);
        this->horizon_dir_2d = horizon_dir / std::abs(horizon_dir);
    }

    virtual std::vector<Complex> for_bbox(const Complex & corner_a, const Complex & corner_b) override {
        (void)corner_a;
        (void)corner_b;
        std::vector<Complex> result;
        if (this->is_inf) {
            return result;
        }
        Complex start = this->projection->internal_position_to_model(
                            this->horizon_anchor_pos
                        );
        Complex direction = this->horizon_dir_2d * this->projection->get_rotation() * this->projection->get_size();
        result.push_back(start - direction);
        result.push_back(start);
        result.push_back(start + direction);
        return result;
    }
};
class HorizonLineCurvilinear : public HorizonLineBase {
public:
    HorizonLineCurvilinear(BaseProjection * projection, const Quaternion & up) {
        (void) projection;
        (void) up;
        // TODO implement
    }
    virtual std::vector<Complex> for_bbox(const Complex & corner_a, const Complex & corner_b) override {
        (void) corner_a;
        (void) corner_b;
        // TODO implement
        std::vector<Complex> result;
        return result;
    }
};
}

PerspectiveLine * RectilinearProjection::get_line(const VanishingPoint& vp, const Complex& start_position) {
    return new PerspectiveLineSimple(this, vp, start_position);
}

HorizonLineBase * RectilinearProjection::get_horizon_line(const Quaternion& up) {
    return new HorizonLineRectilinear(this, up);
}

PerspectiveLine * CurvilinearPerspective::get_line(const VanishingPoint& vp, const Complex& start_position) {
    return new PerspectiveLineCurvilinear(this, vp, start_position);
}

HorizonLineBase * CurvilinearPerspective::get_horizon_line(const Quaternion& up) {
    return new HorizonLineCurvilinear(this, up);
}
