#include <array>
#include "visualizer/atengl.h"
#include "filter/nlm.h"
#include "misc/timer.h"

// NOTE
// http://qiita.com/Ushio/items/56a1c34a5a425ab6b0c2
// http://qiita.com/tobira-code/items/018be1c231e66cc5e28e
// http://opencv.jp/opencv2-x-samples/non-local-means-filter

namespace aten {
	static const int kKernel = 5;
	static const int kSupport = 13;
	static const int kHalfKernel = kKernel / 2;
	static const int kHalfSupport = kSupport / 2;

	using Template = std::array<real, 3 * kKernel * kKernel>;

	Template sampleArea(
		const vec4* src,
		int x, int y,
		int width, int height)
	{
		Template ret;

		int count = 0;

		for (int sx = x - kHalfKernel; sx <= x + kHalfKernel; sx++) {
			for (int sy = y - kHalfKernel; sy <= y + kHalfKernel; sy++) {
				int sample_x = sx;
				int sample_y = sy;

				sample_x = std::max(sample_x, 0);
				sample_x = std::min(sample_x, width - 1);

				sample_y = std::max(sample_y, 0);
				sample_y = std::min(sample_y, height - 1);

				auto p = src + (sample_y * width + sample_x);

				ret[count++] = p->r;
				ret[count++] = p->g;
				ret[count++] = p->b;
			}
		}

		return std::move(ret);
	}

	vec3 samplePixel(
		const vec4* src,
		int x, int y,
		int width, int height)
	{
		int sample_x = x;
		int sample_y = y;

		sample_x = std::max(sample_x, 0);
		sample_x = std::min(sample_x, width - 1);

		sample_y = std::max(sample_y, 0);
		sample_y = std::min(sample_y, height - 1);

		auto p = src + (sample_y * width + sample_x);

		auto ret = *p;

		return std::move(ret);
	}

	static real computeDistanceSquared(const Template& a, const Template& b)
	{
		real sum = 0;

		for (int i = 0; i < a.size(); i++) {
			sum += aten::pow(a[i] - b[i], 2);
		}

		sum /= real(a.size());

		return sum;
	}

	static void doNonLocalMeanFilter(
		const vec4* imgSrc,
		int imgW, int imgH,
		vec4* imgDst,
		real param_h,
		real sigma)
	{
		param_h = std::max(0.0001, param_h);
		sigma = std::max(0.0001, sigma);

		const int width = imgW;
		const int height = imgH;

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				auto dst = imgDst + (y * width + x);

				// 注目領域.
				auto focus = sampleArea(imgSrc, x, y, width, height);

				vec3 sum(0, 0, 0);
				real sum_weight = 0;

				for (int sx = x - kHalfSupport; sx <= x + kHalfSupport; ++sx) {
					for (int sy = y - kHalfSupport; sy <= y + kHalfSupport; ++sy) {
						// 相似度を調べる対象領域.
						auto target = sampleArea(imgSrc, sx, sy, width, height);

						// ノルム（相似度）計算.
						auto dist2 = computeDistanceSquared(focus, target);

						// NOTE
						// Z(p) = sum(exp(-max(|v(p) - v(q)|^2 - 2σ^2, 0) / h^2))
						auto arg = -std::max(dist2 - 2 * sigma * sigma, real(0)) / (param_h * param_h);

						auto weight = exp(arg);

						auto pixel = samplePixel(imgSrc, sx, sy, width, height);

						sum += weight * pixel;
						sum_weight += weight;
					}
				}

				auto color = sum / sum_weight;

				*dst = vec4(color, 1);
			}
		}
	}

	void NonLocalMeanFilter::operator()(
		const vec4* src,
		uint32_t width, uint32_t height,
		vec4* dst)
	{
		timer timer;
		timer.begin();

		doNonLocalMeanFilter(
			src,
			width, height,
			dst,
			m_param_h, m_sigma);

		auto elapsed = timer.end();
		AT_PRINTF("NML %f[ms]\n", elapsed);
	}

	/////////////////////////////////////////////////////////

	void NonLocalMeanFilterShader::prepareRender(
		const void* pixels,
		bool revert)
	{
		Blitter::prepareRender(pixels, revert);

		auto hParam_h = getHandle("param_h");
		if (hParam_h >= 0) {
			CALL_GL_API(::glUniform1f(hParam_h, (float)m_param_h));
		}

		auto hSigma = getHandle("sigma");
		if (hSigma >= 0) {
			CALL_GL_API(::glUniform1f(hSigma, (float)m_sigma));
		}

		// TODO
		// 入力テクスチャのサイズはスクリーンと同じ...
		auto hTexel = getHandle("texel");
		if (hTexel >= 0) {
			CALL_GL_API(glUniform2f(hTexel, 1.0f / m_width, 1.0f / m_height));
		}
	}
}
