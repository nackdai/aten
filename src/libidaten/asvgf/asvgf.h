#pragma once

#include "svgf/svgf.h"

namespace idaten
{
    class AdvancedSVGFPathTracing : public SVGFPathTracing {
    public:
        AdvancedSVGFPathTracing() {}
        virtual ~AdvancedSVGFPathTracing() {}

    public:
        bool setBlueNoises(std::vector<aten::texture*>& noises);

        virtual void render(
            const TileDomain& tileDomain,
            int maxSamples,
            int maxBounce) override;

    protected:
        void onGenPath(
            int maxBounce,
            int seed,
            cudaTextureObject_t texVtxPos,
            cudaTextureObject_t texVtxNml) final;

        void onShade(
            cudaSurfaceObject_t outputSurf,
            int width, int height,
            int bounce, int rrBounce,
            cudaTextureObject_t texVtxPos,
            cudaTextureObject_t texVtxNml) final;

        void onGather(
            cudaSurfaceObject_t outputSurf,
            int width, int height,
            int maxSamples) final;

        void onDebug(
            int width, int height,
            cudaSurfaceObject_t outputSurf);

    protected:
        CudaLeyered2DTexture m_bluenoise;
    };
}
