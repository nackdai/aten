#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"
#include "kernel/pt_standard_impl.h"

namespace idaten
{
    class AORenderer : public StandardPT {
    public:
        AORenderer() = default;
        virtual ~AORenderer() {}

    public:
        virtual void render(
            const TileDomain& tileDomain,
            int32_t maxSamples,
            int32_t maxBounce) override;

        virtual void update(
            GLuint gltex,
            int32_t width, int32_t height,
            const aten::CameraParameter& camera,
            const std::vector<aten::GeometryParameter>& shapes,
            const std::vector<aten::MaterialParameter>& mtrls,
            const std::vector<aten::LightParameter>& lights,
            const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
            const std::vector<aten::TriangleParameter>& prims,
            uint32_t advancePrimNum,
            const std::vector<aten::vertex>& vtxs,
            uint32_t advanceVtxNum,
            const std::vector<aten::mat4>& mtxs,
            const std::vector<TextureResource>& texs,
            const EnvmapResource& envmapRsc) override;

        void setEnableProgressive(bool enable)
        {
            m_enableProgressive = enable;
        }
        bool isEnableProgressive() const
        {
            return m_enableProgressive;
        }

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
        virtual void onShadeMiss(
            int32_t width, int32_t height,
            int32_t bounce);

        virtual void onShade(
            int32_t width, int32_t height,
            int32_t bounce, int32_t rrBounce,
            cudaTextureObject_t texVtxPos,
            cudaTextureObject_t texVtxNml);

        virtual void onGather(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height);

    protected:
        bool m_enableProgressive{ false };

        int32_t m_ao_num_rays{ 1 };
        float m_ao_radius{ 1.0f };
    };
}
