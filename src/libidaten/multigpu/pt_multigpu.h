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
            int maxSamples,
            int maxBounce) override final;

        virtual void postRender(int width, int height) override final;

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
