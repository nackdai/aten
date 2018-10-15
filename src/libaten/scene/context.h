#pragma once

#include <vector>
#include <algorithm>
#include <iterator>

#include "geometry/vertex.h"
#include "visualizer/GeomDataBuffer.h"

namespace aten {
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

    private:
        std::vector<aten::vertex> m_vertices;

        aten::GeomVertexBuffer m_vb;
    };
}
