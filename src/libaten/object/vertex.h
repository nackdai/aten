#pragma once

#include <vector>
#include "types.h"
#include "math/vec3.h"

namespace aten
{
	struct vertex {
		vec3 pos;
		vec3 nml;
		vec3 uv;
	};

	class VertexManager {
		static std::vector<int> s_indices;
		static std::vector<vertex> s_vertices;

	private:
		VertexManager() {}
		~VertexManager() {}

	public:
		static void addIndex(int idx)
		{
			s_indices.push_back(idx);
		}
		static void addVertex(const vertex& vtx)
		{
			s_vertices.push_back(vtx);
		}

		static int getIndex(int pos)
		{
			AT_ASSERT(pos < s_indices.size());
			return s_indices[pos];
		}
		static const vertex& getVertex(int idx)
		{
			AT_ASSERT(idx < s_vertices.size());
			return s_vertices[idx];
		}

		static const std::vector<int>& getIndices()
		{
			return s_indices;
		}
		static const std::vector<vertex>& getVertices()
		{
			return s_vertices;
		}

		static uint32_t getIndexNum()
		{
			return (uint32_t)s_indices.size();
		}
		static uint32_t getVertexNum()
		{
			return (uint32_t)s_vertices.size();
		}
	};
}
