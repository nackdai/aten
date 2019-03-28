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

    protected:
        CudaLeyered2DTexture m_bluenoise;
    };
}
