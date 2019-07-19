#include "asvgf/asvgf.h"

namespace idaten
{
    bool AdvancedSVGFPathTracing::setBlueNoises(std::vector<aten::texture*>& noises)
    {
        const auto W = noises[0]->width();
        const auto H = noises[0]->height();

        // All noise texture have to be same size.
        {
            for (int i = 1; i < noises.size(); i++) {
                const auto n = noises[i];

                auto _w = n->width();
                auto _h = n->height();

                if (W != _w || H != _h) {
                    AT_ASSERT(false);
                    return false;
                }
            }
        }

        std::vector<const aten::vec4*> data;
        for (const auto n : noises) {
            data.push_back(n->colors());
        }

        m_bluenoise.init(data, W, H);

        return true;
    }

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

        m_visibilityBuffer.init(width * height);
    }

    void AdvancedSVGFPathTracing::render(
        const TileDomain& tileDomain,
        int maxSamples,
        int maxBounce)
    {
#if 0
        CudaGLResourceMapper rscmap(&m_glimg);
        auto outputSurf = m_glimg.bind();

        onDebug(tileDomain.w, tileDomain.h, outputSurf);
#else
        int width = tileDomain.w;
        int height = tileDomain.h;

        int tiledWidth = width / GradientTileSize;
        int tiledHeight = height / GradientTileSize;

        m_rngSeed[0].init(width * height);
        m_rngSeed[1].init(width * height);

        m_gradientIndices.init(tiledWidth * tiledHeight);

        if (m_isGeneratedRngSeed) {
            auto vtxPos = m_vtxparamsPos.bind();
            onForwardProjection(width, height, vtxPos);
        }
        else {
            onInitRngSeeds(width, height);
            m_isGeneratedRngSeed = true;
        }

        SVGFPathTracing::render(tileDomain, maxSamples, maxBounce);
#endif
    }
}