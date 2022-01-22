#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"
#include "kernel/renderer.h"

namespace idaten
{
    class AORenderer : public Renderer {
    public:
#ifdef __AT_CUDA__
        struct Path {
            aten::vec3 throughput;
            aten::vec3 contrib;
            aten::sampler sampler;

            real pdfb;
            int samples;

            bool isHit;
            bool isTerminate;
            bool isSingular;
            bool isKill;
        };
        AT_STATICASSERT((sizeof(Path) % 4) == 0);
#else
        struct Path;
#endif

    public:
        AORenderer() {}
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

        int frame() const
        {
            return m_frame;
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
        virtual void onGenPath(
            int width, int height,
            int sample, int maxSamples,
            cudaTextureObject_t texVtxPos,
            cudaTextureObject_t texVtxNml);

        virtual void onHitTest(
            int width, int height,
            cudaTextureObject_t texVtxPos);

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
            idaten::TypedCudaMemory<idaten::AORenderer::Path>& paths,
            int width, int height);

    protected:
        idaten::TypedCudaMemory<idaten::AORenderer::Path> m_paths;
        idaten::TypedCudaMemory<aten::Intersection> m_isects;
        idaten::TypedCudaMemory<aten::ray> m_rays;

        idaten::TypedCudaMemory<int> m_hitbools;
        idaten::TypedCudaMemory<int> m_hitidx;

        idaten::TypedCudaMemory<unsigned int> m_sobolMatrices;
        idaten::TypedCudaMemory<unsigned int> m_random;

        uint32_t m_frame{ 1 };

        bool m_enableProgressive{ false };

        idaten::TileDomain m_tileDomain;

        int m_ao_num_rays{ 1 };
        float m_ao_radius{ 1.0f };
    };
}
