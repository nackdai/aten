#pragma once

#include "renderer/pathtracing.h"
#include "scene/scene.h"
#include "camera/camera.h"

namespace aten
{
	class ERPT : public PathTracing {
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
	};
}
