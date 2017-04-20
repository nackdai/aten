#pragma once

#include "aten.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

	void renderRayTracing(
		aten::vec4* image,
		int width, int height);

#ifdef __cplusplus
}
#endif /* __cplusplus */
