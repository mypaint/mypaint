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

#include "Graph.h"
#include "RawData.h"

struct _object;
typedef _object PyObject;

PyObject * raw_data_to_python(RawGraph & data);
RawGraph python_to_raw_data(PyObject * data);

class PythonGraph : public GraphBase {
public:
    NodeWrapper * create_from_structure(PyObject * data) {
        RawGraph rawGraph = python_to_raw_data(data);
        return GraphBase::create_from_structure(rawGraph);
    }
    NodeWrapper * add_sub_graph(PyObject * data) {
        RawGraph rawGraph = python_to_raw_data(data);
        return GraphBase::add_sub_graph(rawGraph);
    }
    void initialize_from_structure(PyObject * data) {
        RawGraph rawGraph = python_to_raw_data(data);
        GraphBase::initialize_from_structure(rawGraph);
    }
    PyObject * to_object() {
        RawGraph graph = GraphBase::to_raw_data();
        return raw_data_to_python(graph);
    }
};
