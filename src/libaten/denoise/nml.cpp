#include <array>
#include "denoise/nml.h"
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
		const vec3* src,
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
		const vec3* src,
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

		return sum;
	}

	static void doNonLocalMeanFilter(
		const vec3* imgSrc,
		int imgW, int imgH,
		vec3* imgDst,
		real param_h,
		real sigma)
	{
		param_h = std::max(0.0001, param_h);
		sigma = std::max(0.0001, sigma);

		const int width = imgW;
		const int height = imgH;

#pragma omp parallel
		{
#pragma omp for
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
							auto dist = computeDistanceSquared(focus, target);

							// NOTE
							// Z(p) = sum(exp(-max(|v(p) - v(q)|^2 - 2σ^2, 0) / h^2))
							auto arg = -std::max(dist - 2 * sigma * sigma, real(0)) / (param_h * param_h);

							auto weight = exp(arg);

							auto pixel = samplePixel(imgSrc, sx, sy, width, height);

							sum += weight * pixel;
							sum_weight += weight;
						}
					}

					auto color = sum / sum_weight;

					dst->r = color.r;
					dst->g = color.g;
					dst->b = color.b;
				}
			}
		}
	}

	void NonLocalMeanFilter::operator()(
		const vec3* src,
		uint32_t width, uint32_t height,
		vec3* dst)
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
}
