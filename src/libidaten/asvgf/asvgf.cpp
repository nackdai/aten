#include "asvgf/asvgf.h"

namespace idaten
{
    void AdvancedSVGFPathTracing::update(
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
        const EnvmapResource& envmapRsc)
    {
        idaten::SVGFPathTracing::update(
            gltex,
            width, height,
            camera,
            shapes,
            mtrls,
            lights,
            nodes,
            prims, advancePrimNum,
            vtxs, advanceVtxNum,
            mtxs,
            texs, envmapRsc);

        int tiledW = getTiledResolution(width);
        int tiledH = getTiledResolution(height);

        m_gradient.init(tiledW * tiledH);

        m_gradientSample.init(tiledW * tiledH);
    }

    void AdvancedSVGFPathTracing::onDenoise(
        const TileDomain& tileDomain,
        int width, int height,
        cudaSurfaceObject_t outputSurf)
    {
        m_tileDomain = tileDomain;

        if (m_mode == Mode::SVGF
            || m_mode == Mode::TF
            || m_mode == Mode::VAR)
        {
            if (isFirstFrame()) {
                // Nothing is done...
                return;
            }
            else {
                // Create Gradient.
                onCreateGradient(width, height);

                displayTiledData(width, height, m_gradient, outputSurf);

                // TODO
                // Atrous Gradient.

                // TODO
                // Temporal Reprojection.
            }
        }

        if (m_mode == Mode::SVGF
            || m_mode == Mode::VAR)
        {
            onVarianceEstimation(outputSurf, width, height);
        }
    }
}