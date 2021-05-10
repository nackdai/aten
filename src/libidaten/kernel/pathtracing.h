#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"
#include "kernel/renderer.h"

namespace idaten
{
    class PathTracing : public Renderer {
    public:
        struct ShadowRay {
            aten::ray r;
            aten::vec3 lightcontrib;
            real distToLight;
            int targetLightId;

            struct {
                uint32_t isActive : 1;
            };
        };

#ifdef __AT_CUDA__
        struct Path {
            aten::vec3 throughput;
            aten::vec3 contrib;
            aten::sampler sampler;

            real pdfb;
            real accumulatedAlpha;
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
        PathTracing() {}
        virtual ~PathTracing() {}

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

        void update(
            const std::vector<aten::GeomParameter>& geoms,
            const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
            const std::vector<aten::mat4>& mtxs);

        void updateGeometry(
            std::vector<CudaGLBuffer>& vertices,
            uint32_t vtxOffsetCount,
            TypedCudaMemory<aten::PrimitiveParamter>& triangles,
            uint32_t triOffsetCount);

        void updateMaterial(const std::vector<aten::MaterialParameter>& mtrls);

        void enableExportToGLTextures(
            GLuint gltexPosition,
            GLuint gltexNormal,
            GLuint gltexAlbedo,
            const aten::vec3& posRange);

        void enableProgressive(bool enable)
        {
            m_enableProgressive = enable;
        }
        bool isProgressive() const
        {
            return m_enableProgressive;
        }

        int frame() const
        {
            return m_frame;
        }

        enum class AovMode : int {
            None,
            Normal,
            Albedo,
        };

        void setAovMode(AovMode mode)
        {
            aov_mode_ = mode;
        }

        AovMode getAovMode() const
        {
            return aov_mode_;
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
            int sample,
            int bounce, int rrBounce,
            cudaTextureObject_t texVtxPos,
            cudaTextureObject_t texVtxNml);

        virtual void onGather(
            cudaSurfaceObject_t outputSurf,
            idaten::TypedCudaMemory<idaten::PathTracing::Path>& paths,
            int width, int height);

        void displayAov(
            cudaSurfaceObject_t outputSurf,
            int width, int height);

        void copyAovToGLSurface(int width, int height);

    protected:
        idaten::TypedCudaMemory<idaten::PathTracing::Path> m_paths;
        idaten::TypedCudaMemory<aten::Intersection> m_isects;
        idaten::TypedCudaMemory<aten::ray> m_rays;
        idaten::TypedCudaMemory<idaten::PathTracing::ShadowRay> m_shadowRays;

        idaten::TypedCudaMemory<int> m_hitbools;
        idaten::TypedCudaMemory<int> m_hitidx;

        idaten::TypedCudaMemory<unsigned int> m_sobolMatrices;
        idaten::TypedCudaMemory<unsigned int> m_random;

        // For AOV
        AovMode aov_mode_{ AovMode::None };
        idaten::TypedCudaMemory<float4> aov_position_;
        idaten::TypedCudaMemory<float4> aov_nml_;
        idaten::TypedCudaMemory<float3> aov_albedo_;

        // To export to GL.
        bool need_export_gl_{ false };
        aten::vec3 position_range_{ real(1) };
        idaten::TypedCudaMemory<cudaSurfaceObject_t> gl_surface_cuda_rscs_;
        std::vector<idaten::CudaGLSurface> gl_surfaces_;

        uint32_t m_frame{ 1 };

        bool m_enableProgressive{ false };

        idaten::TileDomain m_tileDomain;
    };
}
