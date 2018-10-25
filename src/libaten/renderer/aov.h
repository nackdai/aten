#pragma once

#include "renderer/renderer.h"
#include "scene/scene.h"
#include "camera/camera.h"

namespace aten
{
    class AOVRenderer : public Renderer {
    public:
        AOVRenderer() {}
        virtual ~AOVRenderer() {}

        virtual void onRender(
            const context& ctxt,
            Destination& dst,
            scene* scene,
            camera* camera) override;

    private:
        struct Path {
            vec3 normal;
            vec3 albedo;
            real depth;
            uint32_t shapeid{ 0 };
            uint32_t mtrlid{ 0 };
            uint32_t visibility{ 0 };
        };

        Path radiance(
            const context& ctxt,
            const ray& inRay,
            scene* scene,
            sampler* sampler);

    private:
        uint32_t m_maxDepth{ 1 };
    };
}
