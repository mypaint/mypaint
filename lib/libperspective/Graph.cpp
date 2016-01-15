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
#include <atomic>
#include <iostream>
#include <stack>

#include "Graph.h"
#include "RawData.h"

namespace {
    std::string roleToString(VPRole role) {
        std::string result;
        switch(role) {
            case VPRole::NORMAL:
                result = "NORMAL";
                break;
            case VPRole::SPACE_KEY:
                result = "SPACE";
                break;
        }
        return result;
    }

    std::string nodeRelationToString(NodeRelation relation) {
        std::string result;
        switch (relation) {
            case NodeRelation::CHILD:
                result = "CHILD";
                break;
            case NodeRelation::PARENT:
                result = "PARENT";
                break;
            case NodeRelation::VIEW:
                result = "VIEW";
                break;
            case NodeRelation::COMPUTE:
                result = "COMPUTE";
                break;
            case NodeRelation::COMPUTE_SRC:
                result = "COMPUTE_SRC";
                break;
            case NodeRelation::ANCESTOR_TYPE: [[fallthrough]];
            case NodeRelation::COMPUTE_TYPE:
                throw std::runtime_error("bad node realtion type");
        }
        return result;
    }

    class QuaternionAsVector3 : public Quaternion {
    public:
        explicit QuaternionAsVector3(const Quaternion & q) : Quaternion(q){};
    };

    struct TraverseItem {
        NodeWrapper* parent;
        NodeWrapper* node;
    };

    template<typename F> void forEachNode(NodeWrapper * firstNode, F fct) {
        std::stack<TraverseItem> nodes;
        nodes.push(TraverseItem{
            .parent = nullptr,
            .node = firstNode,
        });
        while (!nodes.empty()) {
            TraverseItem item = nodes.top();
            nodes.pop();
            fct(item.node, item.parent);
            for (auto && child : item.node->get_children()) {
                nodes.push(TraverseItem{
                    .parent = item.node,
                    .node = child,
                });
            }
        }
    }

    CurvilinearPerspective curvilinear_from_raw_data(RawNode & node){
        if (!node.left && !node.right) {
            throw std::runtime_error("bad structure - CurvilinearPerspective");
        }
        return CurvilinearPerspective(*node.left, *node.right);
    }

    RectilinearProjection rectilinear_from_raw_data(RawNode & node) {
        if (!node.left && !node.right) {
            throw std::runtime_error("bad structure - RectilinearProjection");
        }
        return RectilinearProjection(*node.left, *node.right);
    }

    VanishingPoint vp_from_raw_data(RawNode & node){
        if (node.direction) {
            Quaternion direction = *node.direction;
            if (node.direction_local) {
                return VanishingPoint(direction, *node.direction_local);
            } else {
                return VanishingPoint(direction);
            }
        } else if (node.position) {
            return VanishingPoint(*node.position);
        } else {
            throw std::runtime_error("bad structure - VanishingPoint");
        }
    }

    PerspectiveSpace space_from_python(RawNode & node){
        if (!node.rotation && !node.up) {
            std::cout << "missing rotation for perspective space\n";
            std::exit(1);
        } else if (node.rotation && !node.rotation_local) {
            std::cout << "missing local rotation for perspective space\n";
            std::exit(1);
        }

        if (node.up) {
            Quaternion up = *node.up;
            Quaternion defaultUp = Quaternion(0, 1, 0);
            return PerspectiveSpace(rotationBetwenVectors(up, defaultUp));
        } else {
            Quaternion rotation = *node.rotation;
            Quaternion rotationLocal = *node.rotation_local;
            return PerspectiveSpace(rotation, rotationLocal);
        }
    }

    PerspectiveGroup group_from_python() {
        PerspectiveGroup group;
        return group;
    }

    Plane plane_from_python() {
        Plane plane;
        return plane;
    }

    std::shared_ptr<NodeWrapper> base_node_from_python(RawNode & node)    {
        std::string type = node.type;
        std::string name;
        if (node.name) {
            name = *node.name;
        }
        if (type == "VP") {
            auto element = vp_from_raw_data(node);
            return std::make_shared<NodeWrapper>(element, name);
        } else if ( type == "RectilinearProjection") {
            auto element = rectilinear_from_raw_data(node);
            return std::make_shared<NodeWrapper>(element, name);
        } else if ( type == "CurvilinearPerspective") {
            auto element = curvilinear_from_raw_data(node);
            return std::make_shared<NodeWrapper>(element, name);
        } else if ( type == "Group") {
            auto element = group_from_python();
            return std::make_shared<NodeWrapper>(element, name);
        } else if ( type == "Plane") {
            auto element = plane_from_python();
            return std::make_shared<NodeWrapper>(element, name);
        } else if ( type == "Space") {
            auto element = space_from_python(node);
            return std::make_shared<NodeWrapper>(element, name);
        } else {
            std::cout << "unknown perspective element type\n";
            std::exit(1);
        }
    }

    void add_edge(NodeWrapper* dstElement, NodeWrapper* srcElement, const std::unique_ptr<std::string> & edgeType){
        if (!edgeType || *edgeType == "CHILD") {
            srcElement->add_child(dstElement);
            dstElement->add_parent(srcElement);
        } else if ( *edgeType == "PARENT") {
            // pass added in CHILD
        } else if ( *edgeType == "VIEW") {
            if (!dstElement->is_view()) {
                throw std::runtime_error("incorrect view relation");
            }
            srcElement->add_view(dstElement);
        } else if ( *edgeType == "COMPUTE_SRC") {
            srcElement->add_relative(dstElement, NodeRelation::COMPUTE_SRC);
        } else if ( *edgeType == "COMPUTE") {
            srcElement->add_relative(dstElement, NodeRelation::COMPUTE);
        } else {
            throw std::runtime_error("unknown edge element type " + *edgeType);
        }
    }
    
    std::shared_ptr<NodeWrapper> node_from_raw_data(RawNode & rawNode) {
        std::shared_ptr<NodeWrapper> node = base_node_from_python(rawNode);

        if (rawNode.is_UI) {
            node->set_UI(*rawNode.is_UI);
        }

        node->enabled = rawNode.enabled;
        node->parent_enabled = rawNode.parent_enabled;
        node->locked = rawNode.locked;
        node->parent_locked = rawNode.parent_locked;

        if (rawNode.is_compute) {
            bool is_compute = *rawNode.is_compute;
            node->set_compute(is_compute);
            if (is_compute) {
                std::string fctName = *rawNode.compute_fct;
                node->set_compute_fct_by_name(fctName);
                if (rawNode.compute_params) {
                    auto computeParams = *rawNode.compute_params;
                    if (computeParams.size()) {
                        node->set_compute_additional_params(computeParams[0]);
                    }
                }
            }
        } else {
            node->set_compute(false);
        }

        if (rawNode.type == "VP") {
            if (rawNode.role) {
                std::string role = *rawNode.role;
                if (role == "NORMAL") {
                    node->set_role(VPRole::NORMAL);
                } else if (role == "SPACE") {
                    node->set_role(VPRole::SPACE_KEY);
                } else {
                    throw std::runtime_error("bad VanishingPoint role = " + role);
                }
            }

            if (rawNode.color) {
                node->color = *rawNode.color;
            }
        }

        return node;
    }
}

int NodeWrapper::getNextUID(){
    static std::atomic<int> uid { 0 };
    return ++uid;
}

void NodeWrapper::compute(GraphBase* graph) {
    std::vector<NodeWrapper*> src;
    for (auto && relation : _relations) {
        if (relation.relation == NodeRelation::COMPUTE_SRC) {
            src.push_back(relation.node);
        }
    }
    if (src.size() == 0) {
        return;
    }
    if (_compute == nullptr) {
        return;
    }
    (this->*_compute)(src, _compute_additional_params);
    if (is_view() || is_space()) {
        std::vector<NodeWrapper*> computeNodes = graph->update_groups(this);
        for (auto && computeNode : computeNodes) {
            computeNode->compute(graph);
        }
    } else {
        auto view = get_view();
        if (view) {
            view->update_child(this);
        }
    }
}

std::vector<NodeWrapper *> GraphBase::get_points(NodeWrapper* nodeToDraw) {
    std::stack<NodeWrapper*> nodes;
    nodes.push(nodeToDraw);
    std::vector<NodeWrapper*> result;
    while (!nodes.empty()) {
        NodeWrapper * node = nodes.top();
        nodes.pop();
        if (node->is_point()) {
            result.push_back(node);
        }
        for (auto && child : node->get_children()) {
            nodes.push(child);
        }
    }
    return result;
}

std::vector<NodeWrapper *> GraphBase::get_all_enabled_points(bool skipLocked){
    std::stack<NodeWrapper*> nodes;
    nodes.push(get_root());
    std::vector<NodeWrapper*> result;
    while (!nodes.empty()) {
        NodeWrapper * node = nodes.top();
        nodes.pop();
        if (! node->enabled) {
            continue;
        }
        if (node->locked && skipLocked) {
            continue;
        }
        if (node->is_point()) {
            result.push_back(node);
        }
        for (auto && child : node->get_children()) {
            nodes.push(child);
        }
    }
    return result;
}

std::vector<NodeWrapper *> GraphBase::get_all_nodes(NodeWrapper* parent){
    std::vector<NodeWrapper*> result;
    forEachNode(parent, [&result](NodeWrapper * node, NodeWrapper * parent){
        (void) parent;
        result.push_back(node);
    });
    return result;
}


std::vector<NodeWrapper *> GraphBase::get_logic_children(NodeWrapper* node){
    std::stack<NodeWrapper*> groups;
    groups.push(node);
    std::vector<NodeWrapper*> result;
    while (!groups.empty()) {
        NodeWrapper* group = groups.top();
        groups.pop();
        for (auto && child : group->get_children()) {
            // TODO add comments
            if (child->is_UI_only()) {
                groups.push(child);
            } else {
                result.push_back(child);
            }
        }
    }
    return result;
}

std::vector<NodeWrapper *> GraphBase::update_groups(NodeWrapper* group) {
    std::vector<NodeWrapper*> computeNodes;
    NodeWrapper * view = nullptr;
    NodeWrapper * space = nullptr;
    if (group->is_space()) {
        space = group;
        view = group->get_view();
    } else {
        view = group;
    }
    for (auto && child : get_logic_children(group)) {
        for (auto && toCompute : child->get_compute_children()) {
            computeNodes.push_back(toCompute);
        }
        if (child->is_space()) {
            if (space) {
                space->update_subspace(child);
            }
            auto tmpCN = update_groups(child);
            computeNodes.insert(computeNodes.end(), tmpCN.begin(), tmpCN.end());
        } else if (child->is_view()) {
            // pass
        } else {
            if (child->is_compute()) {
                continue;
            }
            if (child->is_point() && space) {
                space->update_child_dir(child);
            }
            view->update_child(child);
        }
    }
    return computeNodes;
}

void GraphBase::update(NodeWrapper* node, Complex pos) {
    std::stack<NodeWrapper*> computeNodes;
    NodeWrapper * space = find_parent_space(node);
    NodeWrapper * view = node->get_view();
    if (node->is_key() && space != nullptr) {
        Quaternion newDir = view->as_projection()->calc_direction(pos);
        space->update_space(node->as_vanishingPoint(), newDir);
        std::vector<NodeWrapper*> tmpCN = update_groups(space);
        for (auto && tmp : tmpCN) {
            computeNodes.push(tmp);
        }
    } else {
        if (space != nullptr) {
            view->update_child(node, pos);
            space->as_space().move_child_to_space(node->as_vanishingPoint());
        } else {
            view->update_child(node,pos);
        }
        for (auto && toCompute : node->get_compute_children()) {
            computeNodes.push(toCompute);
        }
    }

    while (!computeNodes.empty()) {
        NodeWrapper * computeNode = computeNodes.top();
        computeNodes.pop();
        computeNode->compute(this);
        for (auto && toCompute : computeNode->get_compute_children()) {
            computeNodes.push(toCompute);
        }
    }
}

GraphBase::NewElementData GraphBase::get_group_for_new_element(){
    NodeWrapper * chosen = chosen_point;
    NodeWrapper * group = main_view;
    if (chosen != nullptr) {
        if (chosen->is_grouping()) {
            group = chosen;
        } else {
            group = chosen->get_parent();
        }
    }
    NodeWrapper * space = nullptr;
    NodeWrapper * view;
    if (group->is_view()) {
        view = group;
    } else if (group->is_space()) {
        view = group->get_view();
        space = group;
    } else {
        view = group->get_view();
        space = find_parent_space(group);
    }
    return NewElementData{
        .group = group,
        .view = view,
        .space = space,
    };
}

/** connect sub graph with root in local_rot as child of currently selected element */
NodeWrapper * GraphBase::connect_sub_graph(NodeWrapper* localRoot){
    auto data = get_group_for_new_element();

    if (localRoot->is_point()) {
        data.view->update_child(localRoot, localRoot->get_position());
        if (data.space != nullptr) {
            data.space->as_space().move_child_to_space(localRoot->as_vanishingPoint());
        }
    }

    if (localRoot->is_space()) {
        if (data.space != nullptr) {
            data.space->as_space().move_subspace_to_space(localRoot->as_space());
        }
    }

    data.group->add_child(localRoot);
    localRoot->add_parent(data.group);

    std::stack<NodeWrapper*> nodes;
    nodes.push(localRoot);
    while (!nodes.empty()) {
        NodeWrapper * node = nodes.top();
        nodes.pop();
        if (node->get_view() == nullptr) {
            node->add_view(data.view);
            if (node->is_vanishing_point()) {
                data.view->update_child(node);
            }
        } else if (node->is_vanishing_point()) {
            node->get_view()->update_child(node);
        }
        for (auto && child : node->get_children()) {
            nodes.push(child);
        }
    }
    return localRoot;
}

RawGraph GraphBase::to_raw_data() {
    std::map<int, std::string> tagMap;
    RawGraph result;
    for (auto && tag : tags) {
        tagMap[tag.second->uid] = tag.first;
    }
    forEachNode(get_root(), [&result, &tagMap](NodeWrapper * node, NodeWrapper * parent){
        if (parent) {
            RawEdge childEdge;
            childEdge.src = std::to_string(parent->uid);
            childEdge.dst = std::to_string(node->uid);
            childEdge.type = std::make_unique<std::string>(nodeRelationToString(NodeRelation::CHILD));
            result.edges.push_back(std::move(childEdge));
        }
        for (auto && relation : node->get_relations()) {
            RawEdge relationEdge;
            relationEdge.src = std::to_string(node->uid);
            relationEdge.dst = std::to_string(relation.node->uid);
            relationEdge.type = std::make_unique<std::string>(nodeRelationToString(relation.relation));
            result.edges.push_back(std::move(relationEdge));
        }
        RawNode rawNode;
        rawNode.id = std::to_string(node->uid);
        rawNode.name = std::make_unique<std::string>(node->name);
        rawNode.locked = node->locked;
        rawNode.enabled = node->enabled;
        rawNode.parent_enabled = node->parent_enabled;
        rawNode.parent_locked = node->parent_locked;
        rawNode.is_compute = std::make_unique<bool>(node->is_compute());
        rawNode.color = std::make_unique<unsigned>(node->color);
        if (node->is_compute()) {
            rawNode.compute_params = std::make_unique<std::vector<precission>>(node->_compute_additional_params);
            rawNode.compute_fct = std::make_unique<std::string>(node->compute_function_name);
        }

        if (tagMap.count(node->uid)) {
            rawNode.tag = std::make_unique<std::string>(tagMap[node->uid]);
        }

        if (node->_is_group) {
            rawNode.type = "Group";
        } else if (node->is_point()) {
            rawNode.type = "VP";
            rawNode.role = std::make_unique<std::string>(roleToString(node->role));
            rawNode.direction = std::make_unique<Quaternion>(QuaternionAsVector3(node->as_vanishingPoint().get_direction()));
            rawNode.direction_local = std::make_unique<Quaternion>(QuaternionAsVector3(node->as_vanishingPoint().get_direction_local()));
        } else if (node->is_projection()) {
            if (node->_is_curvilinear) {
                rawNode.type = "CurvilinearPerspective";
            } else {
                rawNode.type = "RectilinearProjection";
            }
            rawNode.right = std::make_unique<Complex>(node->as_projection()->calc_pos_from_dir(Quaternion(1, 0, 1, 0)));
            rawNode.left = std::make_unique<Complex>(node->as_projection()->calc_pos_from_dir(Quaternion(-1, 0, 1, 0)));
        } else if (node->is_space()) {
            rawNode.type = "Space";
            rawNode.is_UI = std::make_unique<int>(1);
            rawNode.rotation =  std::make_unique<Quaternion>(node->as_space().get_rotation());
            rawNode.rotation_local =  std::make_unique<Quaternion>(node->as_space().get_rotation_local());
        } else if (node->_is_plane) {
            rawNode.type = "Plane";
        } else {
            throw std::runtime_error("unknown node type");
        }
        result.nodes.push_back(std::move(rawNode));
    });
    
    if (this->visualizations.size()) {
        result.visualizations = std::make_unique<std::vector<RawVisualization>>();
    }

    for (auto && visualization : this->visualizations) {
        RawVisualization visualizationData;
        visualizationData.type = visualization.type;
        for (auto && node : visualization.nodes) {
            visualizationData.nodes.push_back(std::to_string(node));
        }
        result.visualizations->push_back(visualizationData);
    }
    
    result.root =  std::to_string(get_root()->uid);
    result.version = std::make_unique<std::string>("0.3.0");
    return result;
}

NodeWrapper * GraphBase::create_from_structure(RawGraph& data){
    _is_empty = false;
    
    if (data.version) {
        std::string version = *data.version;
        if (*data.version != "0.3.0") {
            // TODO better version checking
            throw std::runtime_error("unsupported version: " + version);
        }
    }
    
    if (!(data.nodes.size() && data.root.size())){
        throw std::runtime_error("bad structure - Graph");
    }
    
    const std::string & root = data.root;
    std::map<std::string, int> idMap;
    
    for (auto && rawNode : data.nodes) {
        std::string dataId = rawNode.id;
        std::shared_ptr<NodeWrapper> node = node_from_raw_data(rawNode);
        idMap[dataId] = node->uid;
        
        this->nodes.push_back(node);
        this->nodeMap[node->uid] = node.get();
        if (rawNode.tag) {
            this->tags[*rawNode.tag] = node.get();
        }
        
        if ((rawNode.type == "RectilinearProjection" || rawNode.type == "CurvilinearPerspective") && this->main_view == nullptr) {
            this->main_view = node.get();
        }
    }
    
    for (auto && rawEdge : data.edges) {
        add_edge( get_by_uid(idMap[rawEdge.dst]), get_by_uid(idMap[rawEdge.src]), rawEdge.type );
    }
    
    if (data.visualizations) {
        for (auto && rawVis : *data.visualizations) {
            VisualizationData visualization;
            visualization.type = rawVis.type;
            for(auto && nodeId : rawVis.nodes) {
                visualization.nodes.push_back(idMap[nodeId]);
            }
            this->visualizations.push_back(visualization);
        }
    }

    return get_by_uid(idMap[root]);
}
