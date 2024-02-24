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
#include "Point.h"

class PerspectiveSpace {
private:
    Quaternion rotation;
    Quaternion rotation_local;
public:
    explicit PerspectiveSpace(const Quaternion & rotation) {
        this->rotation = rotation;
        this->rotation_local = rotation;
    }
    PerspectiveSpace(const Quaternion & rotation, const Quaternion & rotation_local) {
        this->rotation = rotation;
        this->rotation_local = rotation_local;
    }

    /** projection done in not transformed view space */
    Quaternion project_on_space_plane(const Quaternion & direction) {
        Quaternion normalizedDir = normalize(direction);

        Quaternion up = Quaternion(0, 1, 0, 0);
        Quaternion rotation_2 = rotation.conjugate();
        up = rotation * up * rotation_2;

        precission height = up.dot_3D(normalizedDir);
        Quaternion projected = normalizedDir - up.scalar_mul(height);
        return normalize(projected);
    }

    void update_space( const VanishingPoint & vp, const Quaternion & new_direction) {
        Quaternion old_direction = vp.get_direction();

        old_direction = project_on_space_plane(old_direction);
        Quaternion new_direction_projected = project_on_space_plane(new_direction);
        // move to current space
        Quaternion space_rotation = rotation;
        Quaternion space_rotation_2 = space_rotation.conjugate();
        old_direction = space_rotation_2 * old_direction * space_rotation;
        new_direction_projected = space_rotation_2 * new_direction_projected * space_rotation;

        Quaternion rotationTmp = rotationBetwenVectors(old_direction, new_direction_projected);
        rotation = rotation * rotationTmp;
        rotation_local = rotation_local * rotationTmp;
    }

    void update_global_rotation(const Quaternion & new_rotation) {
        rotation = new_rotation;
        rotation_local = rotation_local.conjugate() * new_rotation;
    }

    /** apply rotation on vanishing point */
    void update_child_dir(VanishingPoint & vp) {
        vp.set_direction(rotate(rotation, vp.get_direction_local()));
    }

    /** apply local rotation on subspace */
    void update_subspace(PerspectiveSpace & subspace) {
        subspace.rotation = rotation * subspace.rotation_local;
    }

    void move_child_to_space(VanishingPoint & vp) {
        Quaternion dir_global = vp.get_direction();
        vp.set_direction_local(
            rotate(rotation.conjugate(), dir_global)
        );
    }

    void move_subspace_to_space(PerspectiveSpace & subspace) {
        Quaternion rotation_global = subspace.rotation;
        subspace.rotation_local = rotation.conjugate() * rotation_global;
    }
    
    Quaternion get_rotation() const {
        return rotation;
    }
    
    Quaternion get_rotation_local() const {
        return rotation_local;
    }
};
