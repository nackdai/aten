#pragma once

#include "renderer/renderer.h"
#include "scene/scene.h"
#include "camera/camera.h"

namespace aten
{
	class RayTracing : public Renderer {
	public:
		RayTracing() {}
		virtual ~RayTracing() {}

		virtual void render(
			Destination& dst,
			scene* scene,
			camera* camera) override;
	};
}
