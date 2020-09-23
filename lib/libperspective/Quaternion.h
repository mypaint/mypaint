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
#include <cmath>
#include <sstream>
#include <complex>

using precission = double;
using Complex = std::complex<precission>;

class Quaternion {
public:
    precission x;
    precission y;
    precission z;
    precission w;
    Quaternion() = default;
    constexpr Quaternion(precission x, precission y, precission z) : x(x), y(y), z(z), w(0) {}
    constexpr Quaternion(precission x, precission y, precission z, precission w) : x(x), y(y), z(z), w(w) {}
    constexpr Quaternion operator*(const Quaternion & q) const {
        return Quaternion{
            + this->x * q.w + this->y * q.z - this->z * q.y + this->w * q.x,
            - this->x * q.z + this->y * q.w + this->z * q.x + this->w * q.y,
            + this->x * q.y - this->y * q.x + this->z * q.w + this->w * q.z,
            - this->x * q.x - this->y * q.y - this->z * q.z + this->w * q.w
        };
    }
    constexpr Quaternion operator*(precission scale) const {
        return Quaternion{
            this->x * scale,
            this->y * scale,
            this->z * scale,
            this->w * scale,
        };
    }
    constexpr Quaternion scalar_mul(precission scale) const {
        return Quaternion{
            this->x * scale,
            this->y * scale,
            this->z * scale,
            this->w * scale,
        };
    }
    constexpr Quaternion operator+(const Quaternion & q) const {
        return Quaternion{
            this->x + q.x,
            this->y + q.y,
            this->z + q.z,
            this->w + q.w,
        };
    }
    constexpr Quaternion operator-(const Quaternion & q) const {
        return Quaternion{
            this->x - q.x,
            this->y - q.y,
            this->z - q.z,
            this->w - q.w,
        };
    }
    // TODO tests
    // TODO rename?
    constexpr precission dot_3D(const Quaternion & b) const {
        return this->x * b.x + this->y * b.y + this->z * b.z;
    }
    constexpr Quaternion conjugate() const {
        return { -this->x, -this->y, -this->z, this->w };
    }
    // TODO rename
    char* __str__ () const {
        static char tmp[1024];
        sprintf(tmp, "Q(%g, %g, %g, %g)", x, y, z, w);
        return tmp;
    }
    static constexpr Quaternion FORWARD() {
        return {0, 0, 1, 0};
    }
};

inline precission length(const Quaternion & q) {
    return std::hypot(std::hypot(q.x, q.y), q.z);
}

inline Quaternion normalize(const Quaternion & q) {
    precission len = length(q);
    return {
        q.x/len,
        q.y/len,
        q.z/len,
        q.w
    };
}

constexpr Quaternion conjugate(const Quaternion & q) {
    return { -q.x, -q.y, -q.z, q.w };
}

constexpr Quaternion rotate(const Quaternion & rotation, const Quaternion & vector) {
    return rotation * vector * conjugate(rotation);
}

constexpr Quaternion cross(const Quaternion & a, const Quaternion & b) {
    return Quaternion{
        + a.y * b.z - a.z * b.y,
        - a.x * b.z + a.z * b.x,
        + a.x * b.y - a.y * b.x,
        0
    };
}

constexpr precission dot(const Quaternion & a, const Quaternion & b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Quaternion rotationBetwenVectors(const Quaternion & a, const Quaternion & b) {
    auto normalizedA = normalize(a);
    auto normalizedB = normalize(b);
    auto halfVector = normalize(Quaternion(
        normalizedA.x + normalizedB.x,
        normalizedA.y + normalizedB.y,
        normalizedA.z + normalizedB.z,
        0
    ));
    auto crossVector = cross(normalizedA, halfVector);
    auto cos = dot(normalizedA, halfVector);
    return Quaternion(
        crossVector.x,
        crossVector.y,
        crossVector.z,
        cos
    );
}

// TODO tests
inline Quaternion createRotationQuatenion(const Quaternion & normal, precission angle) {
    precission sin = std::sin(angle/2.0f);
    precission cos = std::cos(angle/2.0f);
    return Quaternion (
        sin * normal.x,
        sin * normal.y,
        sin * normal.z,
        cos
    );
}
