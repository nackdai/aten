#pragma once

#include "renderer/renderer.h"
#include "scene/scene.h"
#include "camera/camera.h"

namespace aten
{
	class ERPT : public Renderer {
	public:
		ERPT() {}
		~ERPT() {}

		virtual void render(
			Destination& dst,
			scene* scene,
			camera* camera) override final;

	private:
		struct Path {
			int x{ 0 };
			int y{ 0 };
			vec3 contrib;
			bool isTerminate{ false };
		};

		Path genPath(
			scene* scene,
			sampler* sampler,
			int x, int y,
			int width, int height,
			camera* camera,
			bool willImagePlaneMutation);

		Path radiance(
			sampler* sampler,
			const ray& inRay,
			camera* cam,
			CameraSampleResult& camsample,
			scene* scene);

	private:
		uint32_t m_maxDepth{ 1 };

		// Depth to compute russinan roulette.
		uint32_t m_rrDepth{ 1 };
	};
}
