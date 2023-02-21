#pragma once

#include "cuda/multigpu.h"
#include "multigpu/renderer_multigpu.h"
#include "kernel/pathtracing.h"

namespace idaten
{
    class PathTracingMultiGPU : public RendererMultiGPU<PathTracing> {
        friend class GpuProxy<PathTracingMultiGPU>;

    public:
        PathTracingMultiGPU() {}
        virtual ~PathTracingMultiGPU() {}

    public:
        virtual void render(
            const TileDomain& tileDomain,
            int32_t maxSamples,
            int32_t maxBounce) override final;

        virtual void postRender(int32_t width, int32_t height) override final;

    protected:
        void setStream(cudaStream_t stream)
        {
            // TODO
        }

    protected:
        void copy(
            PathTracingMultiGPU& from,
            cudaStream_t stream);
    };
}
