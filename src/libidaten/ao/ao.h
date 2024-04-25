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

        virtual void update(
            GLuint gltex,
            int32_t width, int32_t height,
            const aten::CameraParameter& camera,
            const std::vector<aten::ObjectParameter>& shapes,
            const std::vector<aten::MaterialParameter>& mtrls,
            const std::vector<aten::LightParameter>& lights,
            const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
            const std::vector<aten::TriangleParameter>& prims,
            uint32_t advancePrimNum,
            const std::vector<aten::vertex>& vtxs,
            uint32_t advanceVtxNum,
            const std::vector<aten::mat4>& mtxs,
            const std::vector<TextureResource>& texs,
            const aten::BackgroundResource& bg_resource) override;

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

        void onShade(
            int32_t width, int32_t height,
            int32_t bounce, int32_t rrBounce);

    protected:
        int32_t m_ao_num_rays{ 1 };
        float m_ao_radius{ 1.0f };
    };
}
