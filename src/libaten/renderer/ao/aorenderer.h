#pragma once

#include "camera/camera.h"
#include "geometry/EvaluateHitResult.h"
#include "light/pointlight.h"
#include "renderer/renderer.h"
#include "renderer/pathtracing/pt_params.h"
#include "sampler/cmj.h"
#include "scene/scene.h"

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
        void radiance(
            int32_t idx,
            uint32_t rnd,
            const context& ctxt,
            const ray& inRay,
            scene* scene);

    private:
        PathHost path_host_;

        uint32_t m_numAORays{ 1 };
        real m_AORadius{ real(1) };

        int32_t max_depth_{ 1 };
    };
}
