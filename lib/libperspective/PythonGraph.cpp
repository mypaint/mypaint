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

#include "PythonGraph.h"
#include <Python.h>
// FIXME python reference leak

namespace {
    std::string python_to_string(PyObject * obj) {
        PyObject * objStr = PyObject_Str(obj);
#if PYTHON_ABI_VERSION == 3
        return std::string(PyUnicode_AsUTF8(objStr));
#else
        return std::string(PyString_AsString(objStr));
#endif
    }
    
    std::string python_type_string(PyObject * obj) {
        PyObject * type = PyObject_Type(obj);
        return python_to_string(type);
    }
    
    struct pyDictReader;

    struct pyListReader {
    private:
        PyObject * list;
    public:
        explicit pyListReader(PyObject * list) {
            if (PyList_Check(list)) {
                this->list = list;
            } else {
                throw std::runtime_error("python list reader - bad list");
            }
        }
        template<class T, class F> void forAll(F fct) {
            int size = getSize();
            for (int i = 0; i < size; i++) {
                auto value = get<T>(i);
                fct(value);
            }
        }
        template<class T> T get (int id);
        std::string get_as_string(int id) {
            PyObject * item = getItem(id);
            return python_to_string(item);
        }
        pyDictReader getDict(int id);
        int getSize() {
            return PyList_Size(list);
        }
    private:
        PyObject * getItem(int id) {
            PyObject * item = PyList_GetItem(list, id);
            if (item == nullptr) {
                throw std::runtime_error("no list item with id = '"+std::to_string(id)+"'");
            }
            return item;
        }
    };
    template<> precission pyListReader::get(int id){
        return PyFloat_AsDouble(getItem(id));
    }
    template<> std::string pyListReader::get(int id){
        PyObject * item = getItem(id);
        if (!PyUnicode_Check(item)) {
            std::cout << "bad type: expected string for list item " << id << "\n";
            std::cout << " \tget type: " << python_type_string(item) << "\n";
        }
        return python_to_string(item);
    }
    
    struct pyDictReader {
    private:
        PyObject * dict;
    public:
        explicit pyDictReader(PyObject * dict) {
            if (PyDict_Check(dict)) {
                this->dict = dict;
            } else {
                throw std::runtime_error("python dictionary reader - bad dict");
            }
        }
        bool has(const std::string & key) {
            auto pyKey = PyUnicode_FromStringAndSize(key.data(), key.size());
            PyObject * item = PyDict_GetItem(dict, pyKey);
            return item != Py_None && item != nullptr;
        }
        template<class T> T get (const std::string & key);
        pyListReader getList(const std::string & key) {
            PyObject * item = getItem(key);
            if (!PyList_Check(item)) {
                std::cout << "bad type: expected list for dict key " << key << "\n";
                std::cout << " \tget type: " << python_type_string(item) << "\n";
            }
            return pyListReader(item);
        }
        std::string as_string (const std::string & key) {
            PyObject * item = getItem(key);
            return python_to_string(item);
        }
        PyObject * get_object() const {
            return dict;
        }
    private:
        PyObject * getItem(const std::string & key) {
            auto pyKey = PyUnicode_FromStringAndSize(key.data(), key.size());
            PyObject * item = PyDict_GetItem(dict, pyKey);
            if (item == nullptr) {
                throw std::runtime_error("no key - '"+key+"'");
            }
            return item;
        }
    };
    template<> int pyDictReader::get(const std::string& key){
        PyObject * item = getItem(key);
        bool isOk;
#if PYTHON_ABI_VERSION == 3
        isOk = PyLong_Check(item);
#else
        isOk = PyInt_Check(item) || PyLong_Check(item);
#endif
        if (!isOk) {
            std::cout << "bad type: expected Int for dict key " << key << "\n";
            std::cout << " \tget type: " << python_type_string(item) << "\n";
        }
        return PyLong_AsLong(item);
    }
    template<> precission pyDictReader::get(const std::string& key){
        return PyFloat_AsDouble(getItem(key));
    }
    template<> bool pyDictReader::get(const std::string& key){
        return PyObject_IsTrue(getItem(key));
    }
    template<> std::string pyDictReader::get(const std::string& key){
        PyObject * item = getItem(key);
        if (!PyUnicode_Check(item)) {
            std::cout << "bad type: expected string for dicte key " << key << "\n";
            std::cout << " \tget type: " << python_type_string(item) << "\n";
        }
        return python_to_string(item);
    }
    template<> Complex pyDictReader::get(const std::string& key){
        PyObject * item = getItem(key);
        auto getItem = PyList_GetItem;
        if (PyList_Check(item)) {
            getItem = PyList_GetItem;
        } else if (PyTuple_Check(item)) {
            getItem = PyTuple_GetItem;
        } else {
            std::cout << "bad type: expected tuple or list (complex number) for dict key " << key << "\n";
            std::cout << " \tget type: " << python_type_string(item) << "\n";
        }
        return Complex (
            PyFloat_AsDouble(getItem(item, 0)),
            PyFloat_AsDouble(getItem(item, 1))
        );
    }
    template<> Quaternion pyDictReader::get(const std::string& key){
        PyObject * item = getItem(key);
        auto getItem = PyList_GetItem;
        auto getSize = PyList_Size;
        if (PyList_Check(item)) {
            getItem = PyList_GetItem;
            getSize = PyList_Size;
        } else if (PyTuple_Check(item)) {
            getItem = PyTuple_GetItem;
            getSize = PyTuple_Size;
        } else {
            std::cout << "bad type: expected tuple or list (quaternion) for dict key " << key << "\n";
            std::cout << " \tget type: " << python_type_string(item) << "\n";
        }
        auto size = getSize(item);
        if (size == 3) {
            return Quaternion (
                PyFloat_AsDouble(getItem(item, 0)),
                PyFloat_AsDouble(getItem(item, 1)),
                PyFloat_AsDouble(getItem(item, 2))
            );
        } else if (size == 4) {
            return Quaternion (
                PyFloat_AsDouble(getItem(item, 0)),
                PyFloat_AsDouble(getItem(item, 1)),
                PyFloat_AsDouble(getItem(item, 2)),
                PyFloat_AsDouble(getItem(item, 3))
            );
        } else {
            std::cout << "bad type: expected 3 or 4 elements in touple for quaternion in dict for key "<< key << "\n";
            return Quaternion();
        }
    }

    pyDictReader pyListReader::getDict(int id){
        return pyDictReader(getItem(id));
    }

    template<class F> void forEachDict( pyListReader & dictReader, F fct) {
        int size = dictReader.getSize();
        for (int i =0; i < size; i++) {
            fct(dictReader.getDict(i));
        }
    }

    void node_from_python(RawNode & rawNode, pyDictReader & nodeData) {

        pyDictReader reader = pyDictReader(nodeData);
        if (reader.has("direction")) {
            rawNode.direction = std::make_unique<Quaternion>(reader.get<Quaternion>("direction"));
        }
        if (reader.has("direction_local")) {
            rawNode.direction_local = std::make_unique<Quaternion>(reader.get<Quaternion>("direction_local"));
        }
        if (reader.has("position")) {
            rawNode.position = std::make_unique<Complex>(reader.get<Complex>("position"));
        }
        if (reader.has("left")) {
            rawNode.left = std::make_unique<Complex>(reader.get<Complex>("left"));
        }
        if (reader.has("right")) {
            rawNode.right = std::make_unique<Complex>(reader.get<Complex>("right"));
        }
        if (reader.has("up")) {
            rawNode.up = std::make_unique<Quaternion>(reader.get<Quaternion>("up"));
        }
        if (reader.has("rotation")){
            rawNode.rotation = std::make_unique<Quaternion>(reader.get<Quaternion>("rotation"));
        }
        if (reader.has("rotation_local")) {
            rawNode.rotation_local = std::make_unique<Quaternion>(reader.get<Quaternion>("rotation_local"));
        }
        
        if (reader.has("is_UI")) {
            rawNode.is_UI = std::make_unique<int>(reader.get<int>("is_UI"));
        }

        if (reader.has("enabled")) {
            rawNode.enabled = reader.get<bool>("enabled");
        }

        if (reader.has("parent_enabled")) {
            rawNode.parent_enabled = reader.get<bool>("parent_enabled");
        }

        if (reader.has("locked")) {
            rawNode.locked = reader.get<bool>("locked");
        }

        if (reader.has("parent_locked")) {
            rawNode.parent_locked = reader.get<bool>("parent_locked");
        }

        if (reader.has("is_compute")) {
            bool is_compute = reader.get<bool>("is_compute");
            rawNode.is_compute = std::make_unique<bool>(is_compute);
        }

        if (reader.has("compute_fct")) {
            rawNode.compute_fct = std::make_unique<std::string>(reader.as_string("compute_fct"));
        }

        if (reader.has("compute_params")) {
            auto computeParams = reader.getList("compute_params");
            rawNode.compute_params = std::make_unique<std::vector<precission>>();
            if (computeParams.getSize()) {
                rawNode.compute_params->push_back(computeParams.get<precission>(0));
            }
        }

        if (reader.has("role")) {
            rawNode.role = std::make_unique<std::string>(reader.as_string("role"));
        }

        if (reader.has("color")) {
            rawNode.color = std::make_unique<unsigned>(reader.get<int>("color"));
        }
    }

    struct pyListWriter {
    private:
        PyObject * list;
    public:
        pyListWriter() {
            list = PyList_New(0);
        }
        void operator()(PyObject * item) {
            PyList_Append(list, item);
        }
        void operator()(long value) {
            PyList_Append(list, PyLong_FromDouble(value));
        }
        void operator()(int value) {
            PyList_Append(list, PyLong_FromDouble(value));
        }
        void operator()(precission value) {
            PyList_Append(list, PyFloat_FromDouble(value));
        }
        void operator()(std::string value) {
            PyList_Append(list, PyUnicode_FromString(value.c_str()));
        }
        PyObject * result() {
            return list;
        }
    };

    class QuaternionAsVector3 : public Quaternion {
    public:
        explicit QuaternionAsVector3(const Quaternion & q) : Quaternion(q){};
    };

    struct pyDictWriter {
    private:
        PyObject * dict = nullptr;
    public:
        explicit pyDictWriter() {
            this->dict = PyDict_New();
        }
        void operator()(const std::string & name,  int i) {
            add(name, PyLong_FromLong(i));
        }
        void operator()(const std::string & name,  unsigned i) {
            add(name, PyLong_FromLong(i));
        }
        void operator()(const std::string & name,  long i) {
            add(name, PyLong_FromLong(i));
        }
        void operator()(const std::string & name, const std::string & s) {
            add(name, PyUnicode_FromStringAndSize(s.data(), s.size()));
        }
        void operator()(const std::string & name, const char * s) {
            add(name, PyUnicode_FromString(s));
        }
        void operator()(const std::string & name, bool v) {
            add(name, PyBool_FromLong(v));
        }
        void operator()(const std::string & name, PyObject * py) {
            add(name, py);
        }
        void operator()(const std::string & name, QuaternionAsVector3 q) {
            auto tuple = PyTuple_New(3);
            PyTuple_SetItem(tuple, 0, PyFloat_FromDouble(q.x));
            PyTuple_SetItem(tuple, 1, PyFloat_FromDouble(q.y));
            PyTuple_SetItem(tuple, 2, PyFloat_FromDouble(q.z));
            add(name, tuple);
        }
        void operator()(const std::string & name, Quaternion q) {
            auto tuple = PyTuple_New(4);
            PyTuple_SetItem(tuple, 0, PyFloat_FromDouble(q.x));
            PyTuple_SetItem(tuple, 1, PyFloat_FromDouble(q.y));
            PyTuple_SetItem(tuple, 2, PyFloat_FromDouble(q.z));
            PyTuple_SetItem(tuple, 3, PyFloat_FromDouble(q.w));
            add(name, tuple);
        }
        void operator()(const std::string & name, Complex c) {
            auto tuple = PyTuple_New(2);
            PyTuple_SetItem(tuple, 0, PyFloat_FromDouble(c.real()));
            PyTuple_SetItem(tuple, 1, PyFloat_FromDouble(c.imag()));
            add(name, tuple);
        }
        PyObject * result() {
            return dict;
        }
        private:
        void add(const std::string & key, PyObject * value) {
            auto pyKey = PyUnicode_FromStringAndSize(key.data(), key.size());
            PyDict_SetItem(dict, pyKey, value);
        }
    };
}

RawGraph python_to_raw_data(PyObject* data) {
    RawGraph rawGraph;
    pyDictReader allData(data);
    if (allData.has("version")) {
        rawGraph.version = std::make_unique<std::string>(allData.as_string("version"));
    }
    
    rawGraph.root = allData.as_string("root");
    pyListReader nodes = allData.getList("nodes");
    
    forEachDict(nodes, [&rawGraph](pyDictReader nodeData){
        RawNode rawNode;
        rawNode.id = nodeData.as_string("id");
        rawNode.type = nodeData.as_string("type");
        if (nodeData.has("name")) {
            rawNode.name = std::make_unique<std::string>(nodeData.as_string("name"));
        }
        node_from_python(rawNode, nodeData);
        if (nodeData.has("tag")) {
            rawNode.tag = std::make_unique<std::string>(nodeData.as_string("tag"));
        }
        rawGraph.nodes.push_back(std::move(rawNode));
    });

    pyListReader edges = allData.getList("edges");
    forEachDict(edges, [&rawGraph](pyDictReader edgeData) {
        if (!edgeData.has("src") || !edgeData.has("dst")) {
            std::cout << "edge format is incorrect\n";
            std::exit(1);
        }
        RawEdge rawEdge;
        rawEdge.src = edgeData.as_string("src");
        rawEdge.dst = edgeData.as_string("dst");
        if (edgeData.has("type")) {
            rawEdge.type = std::make_unique<std::string>(edgeData.as_string("type"));
        }
        rawGraph.edges.push_back(std::move(rawEdge));
    });
    
    if (allData.has("visualizations")) {
        pyListReader visualizations = allData.getList("visualizations");
        rawGraph.visualizations = std::make_unique<std::vector<RawVisualization>>();
        std::vector<RawVisualization> & rawVisualization = *rawGraph.visualizations;
        forEachDict(visualizations, [&rawVisualization](pyDictReader visualizationData) {
            RawVisualization rawVis;
            rawVis.type = visualizationData.as_string("type");
            pyListReader nodes = visualizationData.getList("nodes");
            int nodesSize = nodes.getSize();
            for (int i = 0; i < nodesSize; i++) {
                rawVis.nodes.push_back(nodes.get_as_string(i));
            }
            rawVisualization.push_back(rawVis);
        });
    }
    
    return rawGraph;
}

PyObject * raw_data_to_python(RawGraph& data) {
    pyListWriter nodes;
    pyListWriter edges;
    std::map<int, std::string> tagMap;
    for (auto && rawEdge : data.edges) {
        pyDictWriter edge;
        edge("src", rawEdge.src);
        edge("dst", rawEdge.dst);
        if (rawEdge.type) {
            edge("type", *rawEdge.type);
        }
        edges(edge.result());
    }
    for (auto && rawNode : data.nodes) {
        pyDictWriter writer;
        writer("id", rawNode.id);
        if (rawNode.name) {
            writer("name", *rawNode.name);
        }
        writer("locked", rawNode.locked);
        writer("enabled", rawNode.enabled);
        writer("parent_enabled", rawNode.parent_enabled);
        writer("parent_locked", rawNode.parent_locked);
        if (rawNode.is_compute) {
            writer("is_compute", *rawNode.is_compute);
        }
        if (rawNode.color) {
            writer("color", *rawNode.color);
        }
        if (rawNode.compute_fct) {
            writer("compute_fct", *rawNode.compute_fct);
        }
        if (rawNode.compute_params) {
            pyListWriter params;
            for (auto && p : *rawNode.compute_params) {
                params(p);
            }
            writer("compute_params", params.result());
        }
        if (rawNode.tag) {
            writer("tag", *rawNode.tag);
        }
        writer("type", rawNode.type);
        if (rawNode.role) {
            writer("role", *rawNode.role);
        }
        if (rawNode.direction) {
            writer("direction", *rawNode.direction);
        }
        if (rawNode.direction_local) {
            writer("direction_local", *rawNode.direction_local);
        }
        if (rawNode.right) {
            writer("right", *rawNode.right);
        }
        if (rawNode.left) {
            writer("left", *rawNode.left);
        }
        if (rawNode.is_UI) {
            writer("is_UI", *rawNode.is_UI);
        }
        if (rawNode.rotation) {
            writer("rotation", *rawNode.rotation);
        }
        if (rawNode.rotation_local) {
            writer("rotation_local", *rawNode.rotation_local);
        }

        nodes(writer.result());
    }

    pyListWriter visualizations;
    if (data.visualizations) {
        for (auto && visualization : *data.visualizations) {
            pyDictWriter visualizationData;
            visualizationData("type", visualization.type);
            pyListWriter visNodes;
            for (auto && node : visualization.nodes) {
                visNodes(node);
            }
            visualizationData("nodes", visNodes.result());
            visualizations(visualizationData.result());
        }
    }

    pyDictWriter result;
    result("nodes", nodes.result());
    result("edges", edges.result());
    result("root", data.root);
    result("visualizations", visualizations.result());
    if (data.version) {
        result("version", *data.version);
    }
    return result.result();
}
