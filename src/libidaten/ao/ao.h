#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"
#include "kernel/pathtracing.h"

namespace idaten
{
    class AORenderer : public PathTracingImplBase {
    public:
        AORenderer() = default;
        virtual ~AORenderer() {}

    public:
        virtual void render(
            int32_t width, int32_t height,
            int32_t maxSamples,
            int32_t maxBounce) override;

        void UpdateSceneData(
            GLuint gltex,
            int32_t width, int32_t height,
            const aten::CameraParameter& camera,
            const aten::context& scene_ctxt,
            const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
            uint32_t advance_prim_num,
            uint32_t advance_vtx_num,
            const aten::BackgroundResource& bg_resource,
            std::function<const aten::Grid* (const aten::context&)> proxy_get_grid_from_host_scene_context = nullptr) override;

        int32_t getNumRays() const
        {
            return m_ao_num_rays;
        }
        void setNumRays(int32_t num)
        {
            m_ao_num_rays = num;
        }

        float getRadius() const
        {
            return m_ao_radius;
        }
        void setRadius(float radius)
        {
            m_ao_radius = radius;
        }

    protected:
        void missShade(
            int32_t width, int32_t height,
            int32_t bounce) override;

        void ShadeAO(
            int32_t width, int32_t height,
            int32_t bounce, int32_t rrBounce);

    protected:
        int32_t m_ao_num_rays{ 1 };
        float m_ao_radius{ 1.0f };
    };
}
