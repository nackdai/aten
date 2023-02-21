#pragma once

#include "renderer/pathtracing.h"
#include "scene/scene.h"
#include "camera/camera.h"
#include "scene/context.h"

namespace aten
{
    // Energy Redistribution Path Tracing
    // http://www.cs.columbia.edu/~batty/misc/ERPT-report.pdf
    class ERPT : public PathTracing {
    public:
        ERPT() {}
        ~ERPT() {}

        virtual void onRender(
            const context& ctxt,
            Destination& dst,
            scene* scene,
            camera* camera) override final;

    private:
        struct Path {
            int32_t x{ 0 };
            int32_t y{ 0 };
            vec3 contrib;
            bool isTerminate{ false };
        };

        Path genPath(
            const context& ctxt,
            scene* scene,
            sampler* sampler,
            int32_t x, int32_t y,
            int32_t width, int32_t height,
            camera* camera,
            bool willImagePlaneMutation);
    };
}
