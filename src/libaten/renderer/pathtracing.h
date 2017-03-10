#pragma once

#include "renderer/renderer.h"
#include "scene/scene.h"
#include "camera/camera.h"

namespace aten
{
	class PathTracing : public Renderer {
	public:
		PathTracing() {}
		~PathTracing() {}

		virtual void render(
			Destination& dst,
			scene* scene,
			camera* camera) override;

	protected:
		struct Path {
			vec3 contrib;
			bool isTerminate{ false };
		};

		Path radiance(
			sampler* sampler,
			const ray& inRay,
			camera* cam,
			CameraSampleResult& camsample,
			scene* scene);

		Path radiance(
			sampler* sampler,
			uint32_t maxDepth,
			const ray& inRay,
			camera* cam,
			CameraSampleResult& camsample,
			scene* scene);

	protected:
		uint32_t m_maxDepth{ 1 };

		// Depth to compute russinan roulette.
		uint32_t m_rrDepth{ 1 };
	};
}
