/*
    This file is part of libPerspective.
    Copyright (C) 2020  Grzegorz Wójcik

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
#include <memory>
#include <vector>
#include <string>
#include "Quaternion.h"

struct RawNode {
    std::string type;
    std::string id;
    std::unique_ptr<std::string> name;
    std::unique_ptr<std::string> tag;
    std::unique_ptr<int> is_UI; // TODO bool?
    std::unique_ptr<bool> is_compute;
    std::unique_ptr<std::string> compute_fct;
    std::unique_ptr<std::vector<precission>> compute_params;
    std::unique_ptr<unsigned> color;
    std::unique_ptr<std::string> role;  // TODO enum?
    std::unique_ptr<Quaternion> direction;
    std::unique_ptr<Quaternion> direction_local;
    std::unique_ptr<Complex> position;
    std::unique_ptr<Complex> left;
    std::unique_ptr<Complex> right;
    std::unique_ptr<Quaternion> up;
    std::unique_ptr<Quaternion> rotation;
    std::unique_ptr<Quaternion> rotation_local;
    bool enabled = true;
    bool parent_enabled = true;
    bool locked = false;
    bool parent_locked = false;
};

struct RawEdge {
    std::string src;
    std::string dst;
    std::unique_ptr<std::string> type;
};

struct RawVisualization {
    std::string type;
    std::vector<std::string> nodes;
};

struct RawGraph {
    std::vector<RawNode> nodes;
    std::vector<RawEdge> edges;
    std::string root;
    std::unique_ptr<std::string> version;
    std::unique_ptr<std::vector<RawVisualization>> visualizations;
};
