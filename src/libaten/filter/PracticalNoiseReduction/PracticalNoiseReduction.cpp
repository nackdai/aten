#include <vector>
#include "filter/PracticalNoiseReduction/PracticalNoiseReduction.h"
#include "filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.h"
#include "misc/color.h"

//#define TEST_filter

#ifdef TEST_filter
#pragma optimize( "", off ) 
#endif

namespace aten {
#if 0
    static inline void _swap(real& v0, real& v1)
    {
        real vMax = std::max(v0, v1);
        real vMin = std::min(v0, v1);

        v0 = vMin;
        v1 = vMax;
    }

    // Median filter.
    static void medianFilter(
        const vec4* src,
        int width, int height,
        vec4* dst)
    {
        // NOTE
        // https://www.gputechconf.jp/assets/files/1017.pdf

        static const int size = 9;    // 3 x 3.
        static const int medianPos = size / 2;

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
        for (int y = 0; y < height; y++) {
            real bufR[size];
            real bufG[size];
            real bufB[size];

            for (int x = 0; x < width; x++) {
                for (int yy = 0; yy < 3; yy++) {
                    for (int xx = 0; xx < 3; xx++) {
                        int bufPos = yy * 3 + xx;
                        
                        int imgx = aten::clamp<int>(x + xx - 1, 0, width - 1);
                        int imgy = aten::clamp<int>(y + yy - 1, 0, height - 1);
                        int imgPos = imgx + imgy * width;

                        bufR[bufPos] = src[imgPos].r;
                        bufG[bufPos] = src[imgPos].g;
                        bufB[bufPos] = src[imgPos].b;
                    }
                }
                
#if 0
                std::sort(bufR, bufR + size);
                std::sort(bufG, bufG + size);
                std::sort(bufB, bufB + size);

                int pos = y * width + x;
                dst[pos] = vec4(bufR[medianPos], bufG[medianPos], bufB[medianPos], 0);
#else
                real maxPixR;
                real maxPixG;
                real maxPixB;

                for (int outer = 0; outer <= medianPos; ++outer) {
                    maxPixR = bufR[outer];
                    maxPixG = bufG[outer];
                    maxPixB = bufB[outer];

                    for (int inner = outer + 1; inner < size; ++inner) {
                        _swap(bufR[inner], maxPixR);
                        _swap(bufG[inner], maxPixG);
                        _swap(bufB[inner], maxPixB);
                    }    
                }

                int pos = y * width + x;
                dst[pos] = vec4(maxPixR, maxPixG, maxPixB, 0);
#endif
            }
        }
    }
#else
    static void gaussianFilter(
        vec4 *_in,
        vec4 *_out,
        int width, int height,
        int filter_size,
        real std_d)
    {
        // ガウスフィルタ係数.
        // g(x) = exp(-1/2 * x^2/d^2) = exp(-(x * x) / (2 * d * d))
        real std_d2 = 2.0f * std_d * std_d;

        int halfWindowSize = filter_size / 2;

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
        for (int cy = 0; cy < height; cy++) {
            for (int cx = 0; cx < width; cx++) {
                int cIdx = cy * width + cx;

                // 計算範囲ウインドウ開始位置.
                int startWindow_x = std::max(0, cx - halfWindowSize);
                int startWindow_y = std::max(0, cy - halfWindowSize);

                // 計算範囲ウインドウ終了位置.
                int endWindow_x = std::min(width - 1, cx + halfWindowSize);
                int endWindow_y = std::min(height - 1, cy + halfWindowSize);

                real sumWeight = 0.0;

                // 出力バッファ初期化.
                _out[cIdx] = vec4(0);

                for (int iy = startWindow_y; iy <= endWindow_y; ++iy) {
                    for (int ix = startWindow_x; ix <= endWindow_x; ++ix) {
                        int idx = iy * width + ix;

                        // ピクセル距離の２乗.
                        real imageDist = (real)(cx - ix) * (cx - ix) + (cy - iy) * (cy - iy);

                        // ガウスフィルタ.
                        // g(x) = exp(-1/2 * x^2/d^2) = exp(-(x * x) / (2 * d * d))
                        real weight = aten::exp(-imageDist / std_d2);

                        _out[cIdx] += weight * _in[idx];

                        sumWeight += weight;
                    }
                }

                _out[cIdx] /= sumWeight;
            }
        }
    }
#endif

    void PracticalNoiseReduction::operator()(
        const vec4* src,
        uint32_t width, uint32_t height,
        vec4* dst)
    {
        AT_PRINTF("PracticalNoiseReduction\n");

        const real stdS = m_stdDevS;
        const real stdC = m_stdDevC;
        const real stdD = m_stdDevD;

        const real t = m_threshold;

#if 1
        std::vector<vec4> prefilter(width * height);
#if 0
        medianFilter(m_indirect, width, height, &prefilter[0]);
#else
        gaussianFilter(
            m_indirect,
            &prefilter[0],
            width, height,
            3,
            stdS);
#endif
        vec4* in = &prefilter[0];
#else
        vec4* in = m_indirect;
#endif

        std::vector<vec4> filtered(width * height);
        std::vector<vec4> var_filtered(width * height);
        std::vector<vec4> hv(width * height);

#if 0
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                real sumW = 0;

                int pos = y * width + x;

                const vec3 p0(x, y, 0);
                vec3 c0 = color::RGBtoXYZ((vec3)in[pos]);
                real d0 = m_nml_depth[pos].w;

                vec4 color;
                vec4 col2;

                static const int Radius = 5;
                static const int HalfRadius = Radius / 2;

                real w[Radius * Radius] = { 0 };
                int cnt = 0;

                for (int yy = -HalfRadius; yy <= HalfRadius; yy++) {
                    for (int xx = -HalfRadius; xx <= HalfRadius; xx++) {
                        int p_x = aten::clamp<int>(x + xx, 0, width - 1);
                        int p_y = aten::clamp<int>(y + yy, 0, height - 1);

                        int p = p_y * width + p_x;

                        const vec4& ci = in[p];
                        vec3 cc = color::RGBtoXYZ((vec3)ci);
                        real di = m_nml_depth[p].w;

                        real l_p = (vec3(p_x, p_y, 0) - p0).length();
                        real l_c = (cc - c0).length();
                        real l_d = di - d0;

                        real g_p = aten::exp(-0.5 * l_p * l_p / (stdS * stdS));
                        real g_c = aten::exp(-0.5 * l_c * l_c / (stdC * stdC));
                        real g_d = aten::exp(-0.5 * l_d * l_d / (stdD * stdD));

                        real weight = g_p * g_c * g_d;

                        color += weight * ci;
                        sumW += weight;

                        w[cnt++] = weight;
                    }
                }

                color /= sumW;

                filtered[pos] = color;

                real weight = 0;
                for (int i = 0; i < cnt; i++) {
                    w[i] /= sumW;
                    weight += w[i] * w[i];
                }
                var_filtered[pos] = weight * m_variance[pos];
            }
        }
#else
        PracticalNoiseReductionBilateralFilter filter;
        filter.setParam(stdS, stdC, stdD);
        filter(
            in, 
            m_nml_depth,
            width, height, 
            &filtered[0],
            &var_filtered[0]);
#endif

#if defined(ENABLE_OMP) && !defined(TEST_filter)
#pragma omp parallel for
#endif
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pos = y * width + x;
                
                const vec4& Lf = filtered[pos];
                const vec4& Llv = m_direct[pos];

                const vec4 Lb = Lf + Llv;
                const vec4 Lb2 = Lb * Lb + vec4(real(0.0001));

                const vec4& varLu = m_variance[pos];
                const vec4& varLf = var_filtered[pos];

                const vec4 u = varLu / Lb2;
                const vec4 f = varLf / Lb2;

                const vec4 D = t * u + t * f - u * f;

                vec4 s;

                if (D.r < 0) {
                    s.r = 0;
                }
                else if (u.r <= t) {
                    s.r = 1;
                }
                else {
                    s.r = (f.r + aten::sqrt(D.r)) / (u.r + f.r);
                }

                if (D.g < 0) {
                    s.g = 0;
                }
                else if (u.g <= t) {
                    s.g = 1;
                }
                else {
                    s.g = (f.g + aten::sqrt(D.g)) / (u.g + f.g);
                }

                if (D.b < 0) {
                    s.b = 0;
                }
                else if (u.b <= t) {
                    s.b = 1;
                }
                else {
                    s.b = (f.b + aten::sqrt(D.b)) / (u.b + f.b);
                }

                s.w = 1;

                //hv[pos] = s * m_indirect[pos] + (vec4(1) - s) * filtered[pos];
                hv[pos] = s * in[pos] + (vec4(1) - s) * filtered[pos];

#if !defined(TEST_filter)
                dst[pos] = m_direct[pos] + hv[pos];
#endif
            }
        }

#ifdef TEST_filter
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pos = y * width + x;

                dst[pos] = m_direct[pos] + hv[pos];
                //dst[pos] = filtered[pos];
            }
        }
#endif
    }
}
