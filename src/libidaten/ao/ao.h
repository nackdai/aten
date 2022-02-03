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
            int maxSamples,
            int maxBounce) override;

        virtual void update(
            GLuint gltex,
            int width, int height,
            const aten::CameraParameter& camera,
            const std::vector<aten::GeomParameter>& shapes,
            const std::vector<aten::MaterialParameter>& mtrls,
            const std::vector<aten::LightParameter>& lights,
            const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
            const std::vector<aten::PrimitiveParamter>& prims,
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

        int getNumRays() const
        {
            return m_ao_num_rays;
        }
        void setNumRays(int num)
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
            int width, int height,
            int bounce);

        virtual void onShade(
            int width, int height,
            int bounce, int rrBounce,
            cudaTextureObject_t texVtxPos,
            cudaTextureObject_t texVtxNml);

        virtual void onGather(
            cudaSurfaceObject_t outputSurf,
            int width, int height);

    protected:
        bool m_enableProgressive{ false };

        int m_ao_num_rays{ 1 };
        float m_ao_radius{ 1.0f };
    };
}
