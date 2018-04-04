#pragma once

#include "material/material.h"

namespace AT_NAME
{
	class FlakesNormal {
	private:
		FlakesNormal();
		~FlakesNormal();

	public:
		static AT_DEVICE_MTRL_API aten::vec4 gen(
			real u, real v,
			real flake_scale = real(50.0),				// Smaller values zoom into the flake map, larger values zoom out.
			real flake_size = real(0.5),				// Relative size of the flakes
			real flake_size_variance = real(0.7),		// 0.0 makes all flakes the same size, 1.0 assigns random size between 0 and the given flake size
			real flake_normal_orientation = real(0.5));	// Blend between the flake normals (0.0) and the surface normal (1.0)
	};
}
