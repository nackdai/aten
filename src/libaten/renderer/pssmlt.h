#pragma once

#include "renderer/pathtracing.h"
#include "scene/scene.h"
#include "camera/camera.h"
#include "scene/context.h"

namespace aten
{
    class PSSMLT : public PathTracing {
    public:
        PSSMLT() {}
        ~PSSMLT() {}

        virtual void onRender(
            const context& ctxt,
            Destination& dst,
            scene* scene,
            camera* camera) override final;

    private:
        struct Path {
            int x{ 0 };
            int y{ 0 };
            vec3 contrib;
            real weight;
            bool isTerminate{ false };
        };

        Path genPath(
            const context& ctxt,
            scene* scene,
            sampler* sampler,
            int x, int y,
            int width, int height,
            camera* camera);
    };
}
