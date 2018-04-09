#pragma once

#include "math/vec4.h"

namespace aten
{
	struct SkinningVertex {
		aten::vec4 position;
		aten::vec3 normal;
		float uv[2];
		float blendIndex[4];
		float blendWeight[4];
	};
}
