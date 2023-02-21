#pragma once

#include "kernel/renderer.h"

namespace idaten
{
    template <class BASE>
    class RendererMultiGPU : public BASE {
    protected:
        RendererMultiGPU() {}
        virtual ~RendererMultiGPU() {}

    public:
        virtual void postRender(int32_t width, int32_t height) = 0;
    };
}
