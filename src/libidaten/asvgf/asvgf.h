#pragma once

#include "svgf/svgf.h"

namespace idaten
{
    class AdvancedSVGFPathTracing : public SVGFPathTracing {
    public:
        static const uint32_t InvalidValue = 0xffffffff;
        static const uint32_t GradientTileSize = 3;
        static const uint32_t StratumOffsetShift = 3;

    public:
        AdvancedSVGFPathTracing() {}
        virtual ~AdvancedSVGFPathTracing() {}

    public:
        bool setBlueNoises(std::vector<aten::texture*>& noises);

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

    protected:
        void onGenPath(
            int maxBounce,
            int seed,
            cudaTextureObject_t texVtxPos,
            cudaTextureObject_t texVtxNml) final;

        void onHitTest(
            int width, int height,
            int bounce,
            cudaTextureObject_t texVtxPos) final;

        void onShade(
            cudaSurfaceObject_t outputSurf,
            int width, int height,
            int bounce, int rrBounce,
            cudaTextureObject_t texVtxPos,
            cudaTextureObject_t texVtxNml) final;

        void onInitRngSeeds(int width, int height);

        void onForwardProjection(
            int width, int height,
            cudaTextureObject_t texVtxPos);

        void onDebug(
            int width, int height,
            cudaSurfaceObject_t outputSurf);

    protected:
        CudaLeyered2DTexture m_bluenoise;

        // xy: bary centric
        // z : object id
        // w : primitive id
        idaten::TypedCudaMemory<float4> m_visibilityBuffer;

        idaten::TypedCudaMemory<float4> m_forwaredWorldCoord;

        idaten::TypedCudaMemory<uint32_t> m_gradientIndices;
        idaten::TypedCudaMemory<uint32_t> m_rngSeed[2];
    };
}
