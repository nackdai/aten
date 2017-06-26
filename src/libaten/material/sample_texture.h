#pragma once

#include "types.h"
#include "math/vec3.h"
#include "texture/texture.h"

namespace AT_NAME {
	inline AT_DEVICE_API aten::vec3 sampleTexture(const int texid, real u, real v, const aten::vec3& defaultValue)
	{
		aten::vec3 ret = defaultValue;

		// TODO
#ifndef __AT_CUDA__
		if (texid >= 0) {
			auto tex = aten::texture::getTexture(texid);
			if (tex) {
				ret = tex->at(u, v);
			}
		}
#endif

		return std::move(ret);
	}
}