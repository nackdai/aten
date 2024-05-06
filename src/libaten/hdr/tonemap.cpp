#include <tuple>
#include <vector>
#include "visualizer/atengl.h"
#include "hdr/tonemap.h"
#include "misc/omputil.h"

namespace aten
{
    // NOTE
    // Reinherd の平均輝度計算.
    // Lavg = exp(1/N * Σlog(δ + L(x, y)))

    // NOTE
    // HDR
    // http://t-pot.com/program/123_ToneMapping/index.html

    std::tuple<float, float> TonemapPreProc::computeAvgAndMaxLum(
        int32_t width, int32_t height,
        const vec4* src)
    {
        auto threadnum = OMPUtil::getThreadNum();
        std::vector<float> sumY(threadnum);
        std::vector<float> maxLum(threadnum);

#ifdef ENABLE_OMP
#pragma omp parallel
#endif
        {
            int32_t cnt = 0;

            auto idx = OMPUtil::getThreadIdx();

#ifdef ENABLE_OMP
#pragma omp for
#endif
            for (int32_t y = 0; y < height; y++) {
                for (int32_t x = 0; x < width; x++) {
                    int32_t pos = y * width + x;

                    const vec3& s = src[pos];

                    vec3 col = vec3(
                        aten::sqrt(s.r),
                        aten::sqrt(s.g),
                        aten::sqrt(s.b));

                    float lum = color::RGBtoY(col);

                    if (lum > float(0)) {
                        sumY[idx] += aten::log(lum);

                        if (lum > maxLum[idx]) {
                            maxLum[idx] = lum;
                        }

                        cnt++;
                    }
                }
            }

            if (cnt > 0) {
                sumY[idx] /= cnt;
            }
        }

        int32_t cnt = 0;
        float retSumY = 0;
        float retMaxLum = 0;

        for (int32_t i = 0; i < threadnum; i++) {
            if (sumY[i] != 0) {
                retSumY += sumY[i];
                cnt++;
            }
            retMaxLum = std::max(maxLum[i], retMaxLum);
        }

        if (cnt > 0) {
            retSumY /= cnt;
        }

        AT_PRINTF("SumY[%f] MaxLum[%f]\n", retSumY, retMaxLum);

        std::tuple<float, float> result = std::make_tuple(retSumY, retMaxLum);
        return result;
    }

    void TonemapPreProc::operator()(
        const vec4* src,
        int32_t width, int32_t height,
        vec4* dst)
    {
        auto result = computeAvgAndMaxLum(
            width, height,
            src);

        auto lum = std::get<0>(result);
        auto maxlum = std::get<1>(result);

        static const float middleGrey = float(0.18);

        const float coeff = middleGrey / aten::exp(lum);
        const float l_max = coeff * maxlum;

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
        for (int32_t h = 0; h < height; h++) {
            for (int32_t w = 0; w < width; w++) {
                int32_t pos = h * width + w;

                const vec3& s = src[pos];
                auto& d = dst[pos];

                vec3 col = vec3(
                    aten::sqrt(s.r),
                    aten::sqrt(s.g),
                    aten::sqrt(s.b));

                vec3 ycbcr = color::RGBtoYCbCr(col);

                ycbcr.y = coeff * ycbcr.y;
                ycbcr.y = ycbcr.y * (1.0f + ycbcr.y / (l_max * l_max)) / (1.0f + ycbcr.y);

                vec3 rgb = color::YCbCrtoRGB(ycbcr);

#if 0
                // Convert to uint8.
                int32_t ir = int32_t(255.9f * rgb.r);
                int32_t ig = int32_t(255.9f * rgb.g);
                int32_t ib = int32_t(255.9f * rgb.b);

                d.r = aten::clamp(ir, 0, 255);
                d.g = aten::clamp(ig, 0, 255);
                d.b = aten::clamp(ib, 0, 255);
#else
                d = vec4(rgb, 1);
#endif
            }
        }
    }

    //////////////////////////////////////////////////////////

    void TonemapPostProc::prepareRender(
        const void* pixels,
        bool revert)
    {
        Blitter::prepareRender(pixels, revert);

        auto result = TonemapPreProc::computeAvgAndMaxLum(width_, height_, (const vec4*)pixels);

        auto lum = std::get<0>(result);
        auto maxlum = std::get<1>(result);

        static const float middleGrey = 0.18f;

        const float coeff = (float)(middleGrey / aten::exp(lum));
        const float l_max = (float)(coeff * maxlum);

        auto hCoeff = getHandle("coeff");
        CALL_GL_API(::glUniform1f(hCoeff, coeff));

        auto hMaxL = getHandle("l_max");
        CALL_GL_API(::glUniform1f(hMaxL, l_max));
    }
}
