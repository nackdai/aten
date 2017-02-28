#include <vector>
#include "denoise/bilateral.h"
#include "misc/timer.h"

// NOTE
// https://github.com/wosugi/compressive-bilateral-filter/blob/master/CompressiveBilateralFilter/original_bilateral_filter.hpp
// http://lcaraffa.net/posts/article-2015TIP-guided-bilateral.html

namespace aten {
	class Sampler {
	public:
		Sampler(vec3* s, uint32_t w, uint32_t h)
			: src(s), width(w), height(h)
		{}
		~Sampler() {}

	public:
		vec3& operator()(int x, int y) const
		{
			x = aten::clamp<int>(x, 0, width - 1);
			y = aten::clamp<int>(y, 0, height - 1);

			auto pos = y * width + x;
			return src[pos];
		}

	private:
		vec3* src;
		uint32_t width;
		uint32_t height;
	};

	// 色距離の重み計算.
	static inline real kernelR(real cdist, real sigmaR)
	{
		auto w = 1.0f / sqrtf(2.0f * AT_MATH_PI * sigmaR) * exp(-0.5f * (cdist * cdist) / (sigmaR * sigmaR));
		return w;
	}

	static inline real kernelS(
		const std::vector<std::vector<real>>& distW,
		int u, int v)
	{
		auto w = distW[v][u];
		return w;
	}

	void doBilateralFilter(
		const vec3* src,
		uint32_t width, uint32_t height,
		real sigmaS, real sigmaR,
		vec3* dst)
	{
		int r = int(::ceilf(4.0f * sigmaS));

		// ピクセル距離の重み.
		std::vector<std::vector<real>> distW;
		{
			distW.resize(1 + r);

			// TODO
			auto _sigmaS = sigmaS * 256;

			for (int v = 0; v <= r; v++) {
				distW[v].resize(1 + r);

				for (int u = 0; u <= r; u++) {
					distW[v][u] = 1.0f / sqrtf(2.0f * AT_MATH_PI * sigmaS) * exp(-0.5f * (u * u + v * v) / (_sigmaS * _sigmaS));
				}
			}
		}

		Sampler srcSampler(const_cast<vec3*>(src), width, height);
		Sampler dstSampler(const_cast<vec3*>(dst), width, height);

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				// 中心点.
				const auto& p = srcSampler(x, y);

				vec3 numer(1.0f, 1.0f, 1.0f);
				vec3 denom(p);

				// (u, 0)
				// 横方向.
				for (int u = 1; u <= r; u++) {
					const auto& p0 = srcSampler(x - u, y);
					const auto& p1 = srcSampler(x + u, y);

					vec3 wr0(
						kernelR(abs(p0.r - p.r), sigmaR),
						kernelR(abs(p0.g - p.g), sigmaR),
						kernelR(abs(p0.b - p.b), sigmaR));
					vec3 wr1(
						kernelR(abs(p1.r - p.r), sigmaR),
						kernelR(abs(p1.g - p.g), sigmaR),
						kernelR(abs(p1.b - p.b), sigmaR));

					numer += kernelS(distW, u, 0) * (wr0 + wr1);
					denom += kernelS(distW, u, 0) * (wr0 * p0 + wr1 * p1);
				}

				// (0, v)
				// 縦方向.
				for (int v = 1; v <= r; v++) {
					const auto& p0 = srcSampler(x, y - v);
					const auto& p1 = srcSampler(x, y + v);

					vec3 wr0(
						kernelR(abs(p0.r - p.r), sigmaR),
						kernelR(abs(p0.g - p.g), sigmaR),
						kernelR(abs(p0.b - p.b), sigmaR));
					vec3 wr1(
						kernelR(abs(p1.r - p.r), sigmaR),
						kernelR(abs(p1.g - p.g), sigmaR),
						kernelR(abs(p1.b - p.b), sigmaR));

					numer += kernelS(distW, 0, v) * (wr0 + wr1);
					denom += kernelS(distW, 0, v) * (wr0 * p0 + wr1 * p1);
				}

				for (int v = 1; v <= r; v++) {
					for (int u = 1; u <= r; u++) {
						const auto& p00 = srcSampler(x - u, y - v);
						const auto& p01 = srcSampler(x - u, y + v);
						const auto& p10 = srcSampler(x + u, y - v);
						const auto& p11 = srcSampler(x + u, y + v);

						vec3 wr00(
							kernelR(abs(p00.r - p.r), sigmaR),
							kernelR(abs(p00.g - p.g), sigmaR),
							kernelR(abs(p00.b - p.b), sigmaR));
						vec3 wr01(
							kernelR(abs(p01.r - p.r), sigmaR),
							kernelR(abs(p01.g - p.g), sigmaR),
							kernelR(abs(p01.b - p.b), sigmaR));
						vec3 wr10(
							kernelR(abs(p10.r - p.r), sigmaR),
							kernelR(abs(p10.g - p.g), sigmaR),
							kernelR(abs(p10.b - p.b), sigmaR));
						vec3 wr11(
							kernelR(abs(p11.r - p.r), sigmaR),
							kernelR(abs(p11.g - p.g), sigmaR),
							kernelR(abs(p11.b - p.b), sigmaR));

						numer += kernelS(distW, u, v) * (wr00 + wr01 + wr10 + wr11);
						denom += kernelS(distW, u, v) * (wr00 * p00 + wr01 * p01 + wr10 * p10 + wr11 * p11);
					}
				}

				auto d = denom / numer;
				dstSampler(x, y) = d;
			}
		}
	}

	void BilateralFilter::operator()(
		const vec3* src,
		uint32_t width, uint32_t height,
		vec3* dst)
	{
		timer timer;
		timer.begin();

		doBilateralFilter(
			src,
			width, height,
			m_sigmaS, m_sigmaR,
			dst);

		auto elapsed = timer.end();
		AT_PRINTF("Bilateral %f[ms]\n", elapsed);
	}
}
