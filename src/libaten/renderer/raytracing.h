#pragma once

#include "renderer/renderer.h"
#include "scene/scene.h"
#include "camera/camera.h"
#include "scene/context.h"

namespace aten
{
    class RayTracing : public Renderer {
    public:
        RayTracing() {}
        virtual ~RayTracing() {}

        virtual void onRender(
            const context& ctxt,
            Destination& dst,
            scene* scene,
            camera* camera) override;

    private:
        vec3 radiance(
            const context& ctxt,
            const ray& ray,
            scene* scene);

    private:
        uint32_t m_maxDepth{ 1 };
    };
}
