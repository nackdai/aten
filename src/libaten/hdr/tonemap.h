#pragma once

#include "math/vec3.h"
#include "misc/color.h"
#include "visualizer/blitter.h"

namespace aten
{
    class TonemapPreProc : public visualizer::PreProc {
    public:
        TonemapPreProc() {}
        virtual ~TonemapPreProc() {}

    public:
        static std::tuple<real, real> computeAvgAndMaxLum(
            int32_t width, int32_t height,
            const vec4* src);

        virtual void operator()(
            const vec4* src,
            uint32_t width, uint32_t height,
            vec4* dst) override final;
    };

    class TonemapPostProc : public Blitter {
    public:
        TonemapPostProc() {}
        virtual ~TonemapPostProc() {}

    public:
        virtual void prepareRender(
            const void* pixels,
            bool revert) override final;
    };
}
