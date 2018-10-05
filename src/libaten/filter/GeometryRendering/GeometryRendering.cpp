#include <vector>
#include "filter/GeometryRendering/GeometryRendering.h"

//#pragma optimize( "", off) 

namespace aten {
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

    struct ColorSasmpler {
        ColorSasmpler(const vec4* s, int w, int h)
            : src(s), width(w), height(h)
        {}

        const vec4& operator()(int x, int y)
        {
            x = aten::clamp(x, 0, width - 1);
            y = aten::clamp(y, 0, height - 1);

            int idx = y * width + x;
            return src[idx];
        }

        const vec4& operator()(int x, int y, int ratio)
        {
            x = aten::clamp(x * ratio, 0, width - 1);
            y = aten::clamp(y * ratio, 0, height - 1);

            int idx = y * width + x;
            return src[idx];
        }

        const vec4* src;
        int width;
        int height;
    };

    void GeometryRendering::getIdx(Idx& idx, const vec4& v)
    {
        idx.shapeid = (uint32_t)v.x;
        idx.mtrlid = (uint32_t)v.y;
    }

    void GeometryRendering::operator()(
        const vec4* src,
        uint32_t width, uint32_t height,
        vec4* dst)
    {
        // NOTE
        // direct, idx は等倍.
        // indirect は 1 / ratio.

        std::vector<vec4> fitelred(width * height);

        const int ratio = m_ratio;

        auto mwidth = width / ratio;
        auto mheight = height / ratio;

        // 参照点位置.
        Pos refPos[4];

        // 参照点ジオメトリインデックス.
        Idx refIdx[4];

        // 参照点色.
        vec4 refClr[4];

        enum {
            UPPER_LEFT,
            LOWER_LEFT,
            LOWER_RIGHT,
            UPPER_RIGHT,
        };

#if 0
        std::vector<vec4> tmp(mwidth * mheight);

        gaussianFilter(
            m_indirect,
            &tmp[0],
            mwidth, mheight,
            3,
            8);

        ColorSasmpler baseSampler(&tmp[0], mwidth, mheight);
#else
        ColorSasmpler baseSampler(m_indirect, mwidth, mheight);
#endif
        ColorSasmpler idxSampler(m_idx, width, height);

        // NOTE
        // +y
        // |
        // |
        // +-----> +x

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int xx = x % ratio;
                int yy = y % ratio;

                int mx = x / ratio;
                int my = y / ratio;

                // 参照点を計算.
                {
                    refPos[UPPER_LEFT].x = mx;
                    refPos[UPPER_LEFT].y = my + 1;

                    refPos[LOWER_LEFT].x = mx;
                    refPos[LOWER_LEFT].y = my;

                    refPos[LOWER_RIGHT].x = mx + 1;
                    refPos[LOWER_RIGHT].y = my;

                    refPos[UPPER_RIGHT].x = mx + 1;
                    refPos[UPPER_RIGHT].y = my + 1;
                }

                const auto& a = baseSampler(refPos[UPPER_LEFT].x, refPos[UPPER_LEFT].y);
                const auto& b = baseSampler(refPos[LOWER_LEFT].x, refPos[LOWER_LEFT].y);
                const auto& c = baseSampler(refPos[LOWER_RIGHT].x, refPos[LOWER_RIGHT].y);
                const auto& d = baseSampler(refPos[UPPER_RIGHT].x, refPos[UPPER_RIGHT].y);

                // 基準点（左下）からの比率を計算.
                real u = aten::abs(x - refPos[LOWER_LEFT].x * ratio) / (real)ratio;
                real v = aten::abs(y - refPos[LOWER_LEFT].y * ratio) / (real)ratio;

                AT_ASSERT(real(0) <= u && u <= real(1));
                AT_ASSERT(real(0) <= v && v <= real(1));

#if 0
                // 一次補間法で計算.
                vec4 l = (1 - u) * v * a + (1 - u) * (1 - v) * b + u * (1 - v) * c + u * v * d;
#else
                u = aten::clamp(u, AT_MATH_EPSILON, real(1));
                v = aten::clamp(v, AT_MATH_EPSILON, real(1));

                getIdx(refIdx[UPPER_LEFT], idxSampler(refPos[UPPER_LEFT].x, refPos[UPPER_LEFT].y, ratio));
                getIdx(refIdx[LOWER_LEFT], idxSampler(refPos[LOWER_LEFT].x, refPos[LOWER_LEFT].y, ratio));
                getIdx(refIdx[LOWER_RIGHT], idxSampler(refPos[LOWER_RIGHT].x, refPos[LOWER_RIGHT].y, ratio));
                getIdx(refIdx[UPPER_RIGHT], idxSampler(refPos[UPPER_RIGHT].x, refPos[UPPER_RIGHT].y, ratio));

                Idx geomIdx;
                getIdx(geomIdx, idxSampler(x, y));

                real norms[4] = {
                    1 / (u * (1 - v)),
                    1 / (u * v),
                    1 / ((1 - u) * v),
                    1 / ((1 - u) * (1 - v)),
                };

                refClr[UPPER_LEFT] = a;
                refClr[LOWER_LEFT] = b;
                refClr[LOWER_RIGHT] = c;
                refClr[UPPER_RIGHT] = d;

                real sumWeight = 0;
                vec4 denom;

                for (int i = 0; i < 4; i++) {
                    int coeff = (refIdx[i].id == geomIdx.id ? 1 : 0);
                    auto weight = norms[i] * coeff;;

                    sumWeight += weight;
                    denom += refClr[i] * weight;
                }

                vec4 l;
                if (sumWeight > 0) {
                    l = denom / sumWeight;
                }
                else {
                    l = (a + b + c + d) / 4;
                }
#endif

                int pos = y * width + x;
                dst[pos] = m_direct[pos] + l;
            }
        }
    }
}
