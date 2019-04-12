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
        void onDebug(
            int width, int height,
            cudaSurfaceObject_t outputSurf);

    protected:
        CudaLeyered2DTexture m_bluenoise;
    };
}
