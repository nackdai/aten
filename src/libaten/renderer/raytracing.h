#pragma once

#include "renderer/destication.h"
#include "scene/scene.h"
#include "camera/camera.h"

namespace aten
{
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
