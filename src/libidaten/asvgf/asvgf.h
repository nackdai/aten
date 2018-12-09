#pragma once

#include "svgf/svgf.h"

namespace idaten
{
    class AsvancedSVGFPathTracing : public SVGFPathTracing {
    public:
        AsvancedSVGFPathTracing() {}
        virtual ~AsvancedSVGFPathTracing() {}

    protected:
        void onCreateGradient();

    protected:
        uint32_t m_tileSize{ 3 };

        TypedCudaMemory<float4> m_gradient[2];
    };
}
