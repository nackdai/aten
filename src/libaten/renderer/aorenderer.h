#pragma once

#include "renderer/renderer.h"
#include "scene/scene.h"
#include "camera/camera.h"
#include "light/pointlight.h"

namespace aten
{
    class AORenderer : public Renderer {
    public:
        AORenderer() {}
        ~AORenderer() {}

        virtual void onRender(
            const context& ctxt,
            Destination& dst,
            scene* scene,
            camera* camera) override;

    protected:
        struct Path {
            vec3 contrib;
            hitrecord rec;
            aten::ray ray;

            Path()
            {
                contrib = vec3(0);
            }
        };

        Path radiance(
            const context& ctxt,
            sampler* sampler,
            const ray& inRay,
            scene* scene);

        bool shade(
            const context& ctxt,
            sampler* sampler,
            scene* scene,
            Path& path);

        void shadeMiss(Path& path);

    private:
        uint32_t m_numAORays{ 1 };
        real m_AORadius{ real(1) };
    };
}
