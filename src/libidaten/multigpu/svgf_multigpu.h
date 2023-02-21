#pragma once

#include "cuda/multigpu.h"
#include "multigpu/renderer_multigpu.h"
#include "svgf/svgf.h"

namespace idaten
{
    class SVGFPathTracingMultiGPU : public RendererMultiGPU<SVGFPathTracing> {
        friend class GpuProxy<SVGFPathTracingMultiGPU>;

    public:
        SVGFPathTracingMultiGPU()
        {
            m_canSSRTHitTest = false;
        }
        virtual ~SVGFPathTracingMultiGPU() {}

    public:
        virtual void render(
            const TileDomain& tileDomain,
            int32_t maxSamples,
            int32_t maxBounce) override final;

        virtual void postRender(int32_t width, int32_t height) override final;

    protected:
        void copy(
            SVGFPathTracingMultiGPU& from,
            cudaStream_t stream);

    private:
        void onRender(
            const TileDomain& tileDomain,
            int32_t width, int32_t height,
            int32_t maxSamples,
            int32_t maxBounce,
            cudaSurfaceObject_t outputSurf,
            cudaTextureObject_t vtxTexPos,
            cudaTextureObject_t vtxTexNml);
    };
}
