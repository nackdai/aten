#pragma once

#include "types.h"
#include "math/vec3.h"
#include "texture/texture.h"

#ifdef __AT_CUDA__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"

namespace AT_NAME {
	AT_DEVICE_MTRL_API aten::vec3 sampleTexture(const int texid, real u, real v, const aten::vec3& defaultValue);

#ifndef __AT_DEBUG__
#include "kernel/sample_texture_impl.cuh"
#endif
}
#else
namespace AT_NAME {
	inline AT_DEVICE_MTRL_API aten::vec3 sampleTexture(const int texid, real u, real v, const aten::vec3& defaultValue)
	{
		aten::vec3 ret = defaultValue;

		// TODO
		if (texid >= 0) {
			auto tex = aten::texture::getTexture(texid);
			if (tex) {
				ret = tex->at(u, v);
			}
		}

		return std::move(ret);
	}
}
#endif