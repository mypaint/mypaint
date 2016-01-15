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
#include "Quaternion.h"
#include <complex>

using precission = double;


class BasePoint {
protected:
    Complex position;
public:
    BasePoint() = default;
    explicit BasePoint(const Complex & position){
        this->position = position; 
    }
    Complex get_position() {
        return position;
    }
    void set_position(const Complex & position) {
        this->position = position;
    }
};


class VanishingPoint : public BasePoint {
private:
    Quaternion dir_local;
    Quaternion direction;
public:
    explicit VanishingPoint(const Complex & pos) : BasePoint(pos) {
    }
    explicit VanishingPoint(const Quaternion & direction) : BasePoint() {
        this->dir_local = direction;
        this->direction = direction;
    }
    VanishingPoint(const Quaternion & direction, const Quaternion & dir_local) : BasePoint() {
        this->dir_local = dir_local;
        this->direction = direction;
    }

    /** return global direction of VP */
    Quaternion get_direction() const {
        return this->direction;
    }
    
    void set_direction(const Quaternion & dir) {
        this->direction = dir;
    }
    
    Quaternion get_direction_local() const {
        return this->dir_local;
    }

    void set_direction_local(const Quaternion & dir) {
        this->dir_local = dir;
    }
};
