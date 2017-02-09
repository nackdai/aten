#pragma once

#include "scene/scene.h"
#include "camera/camera.h"

namespace aten
{
	struct Destination {
		int width{ 0 };
		int height{ 0 };
		uint32_t sample{ 1 };
		uint32_t mutation{ 1 };
		vec3* buffer{ nullptr };
	};

	class RayTracing {
	public:
		RayTracing() {}
		~RayTracing() {}

		void render(
			Destination& dst,
			scene* scene,
			camera* camera);
	};
}
