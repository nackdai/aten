#pragma once

#include "svgf/svgf.h"

namespace idaten
{
    class AdvancedSVGFPathTracing : public SVGFPathTracing {
    public:
        AdvancedSVGFPathTracing() {}
        virtual ~AdvancedSVGFPathTracing() {}

    public:
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
        virtual void onDenoise(
            const TileDomain& tileDomain,
            int width, int height,
            cudaSurfaceObject_t outputSurf) override final;

        void onCreateGradient(int width, int height);

        void displayTiledData(
            int width, int height,
            TypedCudaMemory<float4>& src,
            cudaSurfaceObject_t outputSurf);

        int getTiledResolution(int x) const
        {
            AT_ASSERT(m_gradientTileSize > 0);
            return (x + m_gradientTileSize - 1) / m_gradientTileSize;
        }

    protected:
        int m_gradientTileSize{ 3 };

        TypedCudaMemory<int4> m_gradientSample;
        TypedCudaMemory<float4> m_gradient;

        TypedCudaMemory<int> m_rngSeed[2];
    };
}
