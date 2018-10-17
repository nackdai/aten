#pragma once

#include <vector>
#include <algorithm>
#include <iterator>

#include "geometry/vertex.h"
#include "visualizer/GeomDataBuffer.h"
#include "material/material.h"
#include "misc/datalist.h"

namespace aten
{
class context {
public:
    context() {}
    virtual ~context() {}

public:
    void addVertex(const aten::vertex& vtx)
    {
        m_vertices.push_back(vtx);
    }

    const aten::vertex& getVertex(int idx) const
    {
        return m_vertices[idx];
    }

    aten::vertex& getVertex(int idx)
    {
        return m_vertices[idx];
    }

    const std::vector<aten::vertex>& getVertices() const
    {
        return m_vertices;
    }

    uint32_t getVertexNum() const
    {
        return (uint32_t)m_vertices.size();
    }

    void copyVertices(std::vector<vertex>& dst) const
    {
        std::copy(
            m_vertices.begin(),
            m_vertices.end(),
            std::back_inserter(dst));
    }

    void build();

    const aten::GeomVertexBuffer& getVB() const
    {
        return m_vb;
    }

    void release()
    {
        m_vertices.clear();
        m_vb.clear();
    }

    void addMaterial(AT_NAME::material* mtrl);

    DataList<AT_NAME::material>& getMaterials()
    {
        return m_materials;
    }

    const DataList<AT_NAME::material>& getMaterials() const
    {
        return m_materials;
    }

    uint32_t getMaterialNum() const
    {
        return static_cast<uint32_t>(getMaterials().size());
    }

    const AT_NAME::material* getMaterial(uint32_t idx) const
    {
        AT_ASSERT(idx < getMaterialNum());
        return m_materials[idx];
    }

    void copyMaterialParameters(std::vector<MaterialParameter>& dst) const;

    const AT_NAME::material* findMaterialByName(const char* name) const;

    int findMaterialIdxByName(const char* name) const;

private:
    std::vector<aten::vertex> m_vertices;

    aten::GeomVertexBuffer m_vb;

    DataList<AT_NAME::material> m_materials;
};
}
