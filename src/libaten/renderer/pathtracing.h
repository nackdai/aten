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
			scene* scene,
			camera* camera);
	};
}
