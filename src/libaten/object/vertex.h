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
		static std::vector<vec3> s_positions;
		static std::vector<vec3> s_normals;
		static std::vector<vec3> s_uvs;

	private:
		VertexManager() {}
		~VertexManager() {}

	public:
		static void addIndex(int idx)
		{
			s_indices.push_back(idx);
		}
		static void addPositon(float x, float y, float z)
		{
			s_positions.push_back(vec3(x, y, z));
		}
		static void addNormal(float x, float y, float z)
		{
			s_normals.push_back(vec3(x, y, z));
		}
		static void addUV(float u, float v)
		{
			s_uvs.push_back(vec3(u, v, 0));
		}

		static int getIndex(int pos)
		{
			AT_ASSERT(pos < s_indices.size());
			return s_indices[pos];
		}
		static void getVertex(vertex& vtx, int idx)
		{
			if (idx < s_positions.size()) {
				vtx.pos = s_positions[idx];
			}
			if (idx < s_normals.size()) {
				vtx.nml = s_normals[idx];
			}
			if (idx < s_uvs.size()) {
				vtx.uv = s_uvs[idx];
			}
		}

		static const std::vector<int>& getIndices()
		{
			return s_indices;
		}
		static const std::vector<vec3>& getPositions()
		{
			return s_positions;
		}
		static const std::vector<vec3>& getNormals()
		{
			return s_normals;
		}
		static const std::vector<vec3>& getUVs()
		{
			return s_uvs;
		}

		static uint32_t getPositionNum()
		{
			return (uint32_t)s_positions.size();
		}
	};
}
