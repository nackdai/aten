#pragma once

#include <vector>
#include "types.h"
#include "math/vec4.h"
#include "visualizer/GeomDataBuffer.h"

namespace aten
{
    struct vertex {
        vec4 pos;
        vec3 nml;

        // z == 1, compute plane normal in real-time.
        // z == -1, there is no texture coordinate.
        vec3 uv;
    };

    struct CompressedVertex {
        // NOTE
        // pos.w == uv.x, nml.w == uv.y.
        vec4 pos;
        vec4 nml;
    };

    class VertexManager {
        static std::vector<vertex> s_vertices;

        static GeomVertexBuffer s_vb;

    private:
        VertexManager() {}
        ~VertexManager() {}

    public:
        static void addVertex(const vertex& vtx)
        {
            s_vertices.push_back(vtx);
        }

        static vertex& getVertex(int idx)
        {
            return s_vertices[idx];
        }

        static const std::vector<vertex>& getVertices()
        {
            return s_vertices;
        }

        static uint32_t getVertexNum()
        {
            return (uint32_t)s_vertices.size();
        }

        static void build();
        static GeomVertexBuffer& getVB()
        {
            return s_vb;
        }

        static void release()
        {
            s_vertices.clear();
            s_vb.clear();
        }
    };
}
