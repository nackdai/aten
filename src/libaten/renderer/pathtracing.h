#pragma once

#include "renderer/destication.h"
#include "scene/scene.h"
#include "camera/camera.h"

namespace aten
{
	class PathTracing {
	public:
		PathTracing() {}
		~PathTracing() {}

		void render(
			Destination& dst,
			scene& scene,
			camera* camera);

	private:
		vec3 radiance(
			sampler& sampler,
			const ray& inRay,
			scene& scene);

	private:
		uint32_t m_maxDepth{ 1 };

		// Depth to compute russinan roulette.
		uint32_t m_rrDepth{ 1 };
	};
}
