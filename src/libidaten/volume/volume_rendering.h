#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"
#include "kernel/pathtracing.h"

#include "misc/tuple.h"
#include "sampler/sampler.h"

namespace idaten
{
    class VolumeRendering : public PathTracing {
    public:
        VolumeRendering() {}
        virtual ~VolumeRendering()
        {
            if (m_stream) {
                cudaStreamDestroy(m_stream);
            }
        }

    public:
        std::optional<aten::aabb> LoadNanoVDB(std::string_view nvdb);

        void RenderNanoVDB(
            GLuint gltex,
            int32_t width, int32_t height,
            const aten::CameraParameter& camera,
            const aten::vec3 bg_color = aten::vec3(0.0F, 0.5F, 1.0F));

    protected:
        void onShade(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height,
            int32_t sample,
            int32_t bounce, int32_t rrBounce, int32_t max_depth) override;

        class SimpleGridRenderer;
        std::shared_ptr<SimpleGridRenderer> simple_grid_renderer_;

        cudaStream_t m_stream{ nullptr };
    };
}
