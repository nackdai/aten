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

    DataList<aten::material>& getMaterials()
    {
        return m_materials;
    }

    const DataList<aten::material>& getMaterials() const
    {
        return m_materials;
    }

    uint32_t getMaterialNum() const
    {
        return static_cast<uint32_t>(getMaterials().size());
    }

    const material* getMaterial(uint32_t idx) const
    {
        AT_ASSERT(idx < getMaterialNum());
        return m_materials[idx];
    }

    void clearMaterialList()
    {

    }

    // TODO
    // マテリアルにIDを持たせているので、この関数は不要.
    int findMaterialIdx(material* mtrl);

    int findMaterialIdxByName(const char* name) const;

private:
    std::vector<aten::vertex> m_vertices;

    aten::GeomVertexBuffer m_vb;

    DataList<aten::material> m_materials;
};
}
