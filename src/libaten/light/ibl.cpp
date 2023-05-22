#include "light/ibl.h"
#include "scene/host_scene_context.h"

// NOTE
// http://www.cs.virginia.edu/~gfx/courses/2007/ImageSynthesis/assignments/envsample.pdf
// Importance Sampling for Production Rendering
// http://www.igorsklyar.com/system/documents/papers/4/fiscourse.comp.pdf

namespace AT_NAME {
    void ImageBasedLight::preCompute()
    {
        auto envmap = getEnvMap();
        AT_ASSERT(envmap);

        auto width = envmap->getTexture()->width();
        auto height = envmap->getTexture()->height();

        m_avgIllum = 0;

        // NOTE
        // 環境マップは、正距円筒（= 緯度経度マップ）のみ.

        // NOTE
        //    0 1 2 3 4
        //   +-+-+-+-+-+
        // 0 |a|b|c|d|e|...
        //   +-+-+-+-+-+
        // 1 |h|i|j|k|l|...
        //   +-+-+-+-+-+
        //
        // V方向（縦方向）のPDFは１列分の合計値となる.
        // 　pdfV_0 = a + b + c + d + e + ....
        // 　pdfV_1 = h + i + j + k + l + ....
        //
        // U方向（横方向）のPDFは１ピクセルずつの値となる.
        // 　pdfU_00 = a, pdfU_01 = b, ...
        // 　pdfU_10 = h, pdfU_11 = i, ...

        real totalWeight = 0;

        m_cdfU.resize(height);

        for (uint32_t y = 0; y < height; y++) {
            // NOTE
            // 正距円筒は、緯度方向については極ほど歪むので、その補正.
            // 緯度方向は [0, pi].
            // 0.5 足すのは、ピクセル中心点をサンプルするのと、ゼロにならないようにするため.
            // sin(0) = 0 で scale値がゼロになるのを避けるため.
            real scale = aten::sin(AT_MATH_PI * (real)(y + 0.5) / height);

            // v方向のpdf.
            real pdfV = 0;

            // u方向のpdf.
            std::vector<real>& pdfU = m_cdfU[y];

            for (uint32_t x = 0; x < width; x++) {
                real u = (real)(x + 0.5) / width;
                real v = (real)(y + 0.5) / height;

                auto clr = envmap->sample(u, v);
                const auto illum = AT_NAME::color::luminance(clr);

                m_avgIllum += illum * scale;
                totalWeight += scale;

                // １列分の合計値を計算.
                pdfV += illum * scale;

                // まずはpdfを貯める.
                pdfU.push_back(illum * scale);
            }

            // まずはpdfを貯める.
            m_cdfV.push_back(pdfV);
        }

        // For vertical.
        {
            real sum = 0;
            for (int32_t i = 0; i < m_cdfV.size(); i++) {
                sum += m_cdfV[i];
                if (i > 0) {
                    m_cdfV[i] += m_cdfV[i - 1];
                }
            }
            if (sum > 0) {
                real invSum = 1 / sum;
                for (int32_t i = 0; i < m_cdfV.size(); i++) {
                    m_cdfV[i] *= invSum;
                    m_cdfV[i] = aten::clamp<real>(m_cdfV[i], 0, 1);
                }
            }
        }

        // For horizontal.
        {
            for (uint32_t y = 0; y < height; y++) {
                real sum = 0;
                std::vector<real>& cdfU = m_cdfU[y];

                for (uint32_t x = 0; x < width; x++) {
                    sum += cdfU[x];
                    if (x > 0) {
                        cdfU[x] += cdfU[x - 1];
                    }
                }

                if (sum > 0) {
                    real invSum = 1 / sum;
                    for (uint32_t x = 0; x < width; x++) {
                        cdfU[x] *= invSum;
                        cdfU[x] = aten::clamp<real>(cdfU[x], 0, 1);
                    }
                }
            }
        }

        m_avgIllum /= totalWeight;
    }

    real ImageBasedLight::samplePdf(const aten::ray& r) const
    {
        auto envmap = getEnvMap();

        auto clr = envmap->sample(r);

        auto pdf = samplePdf(clr, m_avgIllum);

        return pdf;
    }

    static int32_t samplePdfAndCdf(
        real r,
        const std::vector<real>& cdf,
        real& outPdf,
        real& outCdf)
    {
        outPdf = 0;
        outCdf = 0;

        // NOTE
        // cdf is normalized to [0, 1].

#if 1
        int32_t idxTop = 0;
        int32_t idxTail = (int32_t)cdf.size() - 1;

        for (;;) {
            int32_t idxMid = (idxTop + idxTail) >> 1;
            auto midCdf = cdf[idxMid];

            if (r < midCdf) {
                idxTail = idxMid;
            }
            else {
                idxTop = idxMid;
            }

            if ((idxTail - idxTop) == 1) {
                auto topCdf = cdf[idxTop];
                auto tailCdf = cdf[idxTail];

                int32_t idx = 0;

                if (r <= topCdf) {
                    outPdf = topCdf;
                    idx = idxTop;
                }
                else {
                    outPdf = tailCdf - topCdf;
                    idx = idxTail;
                }

                return idx;
            }
        }
#else
        for (int32_t i = 0; i < cdf.size(); i++) {
            if (r <= cdf[i]) {
                auto idx = i;

                outCdf = cdf[i];

                if (i > 0) {
                    outPdf = cdf[i] - cdf[i - 1];
                }
                else {
                    outPdf = cdf[0];
                }

                return idx;
            }
        }
#endif

        AT_ASSERT(false);
        return 0;
    }

    aten::LightSampleResult ImageBasedLight::sample(
        const aten::context& ctxt,
        const aten::vec3& org,
        const aten::vec3& nml,
        aten::sampler* sampler) const
    {
        auto envmap = getEnvMap();

        aten::LightSampleResult result;

#if 1
        const auto r1 = sampler->nextSample();
        const auto r2 = sampler->nextSample();

        real pdfU, pdfV;
        real cdfU, cdfV;

        int32_t y = samplePdfAndCdf(r1, m_cdfV, pdfV, cdfV);
        int32_t x = samplePdfAndCdf(r2, m_cdfU[y], pdfU, cdfU);

        auto width = envmap->getTexture()->width();
        auto height = envmap->getTexture()->height();

        real u = (real)(x + 0.5) / width;
        real v = (real)(y + 0.5) / height;

        // NOTE
        // p(w) = p(u, v) * (w * h) / (2π^2 * sin(θ))
        auto pi2 = AT_MATH_PI * AT_MATH_PI;
        auto theta = AT_MATH_PI * v;
        result.pdf = (pdfU * pdfV) * ((width * height) / (pi2 * aten::sin(theta)));

        // u, v -> direction.
        result.dir = AT_NAME::envmap::convertUVToDirection(u, v);

        result.le = envmap->sample(u, v);
        result.finalColor = result.le;
#elif 0
        auto uv = AT_NAME::envmap::convertDirectionToUV(nml);

        static const real radius = real(3);

        auto r1 = sampler->nextSample();
        auto r2 = sampler->nextSample();

        // [0, 1] -> [-1, 1]
        r1 = r1 * 2 - 1;
        r2 = r2 * 2 - 1;

        auto width = envmap->getTexture()->width();
        auto height = envmap->getTexture()->height();

        r1 = r1 * radius / (real)width;
        r2 = r2 * radius / (real)height;

        real u = uv.x + r1;
        if (u < real(0)) {
            u += real(1);
        }
        else if (u > real(1)) {
            u = real(1) - (u - real(1));
        }

        real v = uv.y + r2;
        if (v < real(0)) {
            v += real(1);
        }
        else if (v > real(1)) {
            v = real(1) - (v - real(1));
        }

        // u, v -> direction.
        result.dir = AT_NAME::envmap::convertUVToDirection(u, v);

        result.pdf = dot(nml, result.dir) / AT_MATH_PI;

        result.le = envmap->sample(u, v);
        result.intensity = real(1);
        result.finalColor = result.le * result.intensity;
#else
        auto n = nml;
        auto t = aten::getOrthoVector(nml);
        auto b = normalize(cross(n, t));

        real r1 = sampler->nextSample();
        real r2 = sampler->nextSample();

        real sinpsi = aten::sin(2 * AT_MATH_PI * r1);
        real cospsi = aten::cos(2 * AT_MATH_PI * r1);
        real costheta = aten::pow(1 - r2, 0.5);
        real sintheta = aten::sqrt(1 - costheta * costheta);

        // returnTo the result
        result.dir = normalize(t * sintheta * cospsi + b * sintheta * sinpsi + n * costheta);

        result.pdf = dot(nml, result.dir) / AT_MATH_PI;

        auto uv = AT_NAME::envmap::convertDirectionToUV(result.dir);

        result.le = envmap->sample(uv.x, uv.y);
        result.intensity = real(1);
        result.finalColor = result.le * result.intensity;
#endif

        // TODO
        // Currently not used...
        result.pos = aten::vec3();
        result.nml = aten::vec3();

        return result;
    }
}
