#pragma once

#include <string>
#include <vector>
#include "math/vec3.h"

namespace aten
{
	class HDRExporter {
	public:
		static bool save(
			const std::string& filename,
			const vec3* image,
			const int width, const int height);
	};
}