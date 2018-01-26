#pragma once

#include "aten.h"
#include <vector>

class LodMaker {
public:
	static void make(
		std::vector<aten::vertex>& dstVertices,
		std::vector<std::vector<int>>& dstIndices,
		const aten::aabb& bound,
		const std::vector<aten::vertex>& vertices,
		const std::vector<std::vector<aten::face*>>& tris,
		int gridX,
		int gridY,
		int gridZ);
};