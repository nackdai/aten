#pragma once

#include "aten4idaten.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

	void renderRayTracing(
		aten::vec4* image,
		int width, int height,
		std::vector<aten::LightParameter> lights);

#ifdef __cplusplus
}
#endif /* __cplusplus */
