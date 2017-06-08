#include <vector>
#include "filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.h"
#include "misc/timer.h"

// NOTE
// https://github.com/wosugi/compressive-bilateral-filter/blob/master/CompressiveBilateralFilter/original_bilateral_filter.hpp
// http://lcaraffa.net/posts/article-2015TIP-guided-bilateral.html

//#pragma optimize( "", off )

namespace aten {
	class Sampler {
	public:
		Sampler(vec4* s, uint32_t w, uint32_t h)
			: src(s), width(w), height(h)
		{}
		~Sampler() {}

	public:
		vec4& operator()(int x, int y) const
		{
			x = aten::clamp<int>(x, 0, width - 1);
			y = aten::clamp<int>(y, 0, height - 1);

			auto pos = y * width + x;
			return src[pos];
		}

		void set(int x, int y, const vec4& v)
		{
			if (src) {
				(*this)(x, y) = v;
			}
		}

	private:
		vec4* src;
		uint32_t width;
		uint32_t height;
	};

	// 色距離の重み計算.
	static inline real kernelR(real cdist, real sigmaR)
	{
		//auto w = 1.0f / sqrtf(2.0f * AT_MATH_PI * sigmaR) * exp(-0.5f * (cdist * cdist) / (sigmaR * sigmaR));
		auto w = exp(-0.5f * (cdist * cdist) / (sigmaR * sigmaR));
		return w;
	}

	// 深度の重み計算.
	static inline real kernelD(real ddist, real sigmaD)
	{
		//auto w = 1.0f / sqrtf(2.0f * AT_MATH_PI * sigmaD) * exp(-0.5f * (ddist * ddist) / (sigmaD * sigmaD));
		auto w = exp(-0.5f * (ddist * ddist) / (sigmaD * sigmaD));
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
		const vec4* src,
		const vec4* nml_depth,
		uint32_t width, uint32_t height,
		real sigmaS, real sigmaR, real sigmaD,
		vec4* dst,
		vec4* variance)
	{
		//int r = int(::ceilf(4.0f * sigmaS));
		int r = 3;

		// ピクセル距離の重み.
		std::vector<std::vector<real>> distW;
		{
			distW.resize(1 + r);

			// TODO
			//auto _sigmaS = sigmaS * 256;
			auto _sigmaS = sigmaS;

			for (int v = 0; v <= r; v++) {
				distW[v].resize(1 + r);

				for (int u = 0; u <= r; u++) {
					//distW[v][u] = 1.0f / sqrtf(2.0f * AT_MATH_PI * sigmaS) * exp(-0.5f * (u * u + v * v) / (_sigmaS * _sigmaS));
					distW[v][u] = exp(-0.5f * (u * u + v * v) / (_sigmaS * _sigmaS));
				}
			}
		}

		Sampler srcSampler(const_cast<vec4*>(src), width, height);
		Sampler depthSampler(const_cast<vec4*>(nml_depth), width, height);

		Sampler dstSampler(const_cast<vec4*>(dst), width, height);

		Sampler varSampler(const_cast<vec4*>(variance), width, height);

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				// 中心点.
				const auto& p = srcSampler(x, y);
				const real dc = depthSampler(x, y).w;

				vec3 numer = vec3(1.0f, 1.0f, 1.0f);
				vec3 denom = vec3(p.x, p.y, p.z);

				vec3 denom2 = vec3(p.x * p.x, p.y * p.y, p.z * p.z);

				// (u, 0)
				// 横方向.
				for (int u = 1; u <= r; u++) {
					const auto& p0 = srcSampler(x - u, y);
					const auto& p1 = srcSampler(x + u, y);

					vec3 wr0 = vec3(
						kernelR(abs(p0.r - p.r), sigmaR),
						kernelR(abs(p0.g - p.g), sigmaR),
						kernelR(abs(p0.b - p.b), sigmaR));
					vec3 wr1 = vec3(
						kernelR(abs(p1.r - p.r), sigmaR),
						kernelR(abs(p1.g - p.g), sigmaR),
						kernelR(abs(p1.b - p.b), sigmaR));

					const auto& d0 = depthSampler(x - u, y);
					const auto& d1 = depthSampler(x + u, y);

					const real dd0 = kernelD(d0.w - dc, sigmaD);
					const real dd1 = kernelD(d1.w - dc, sigmaD);

					numer += kernelS(distW, u, 0) * (wr0 * dd0 + wr1 * dd1);
					auto d = kernelS(distW, u, 0) * (wr0 * vec3(p0) * dd0 + wr1 * vec3(p1) * dd1);
					denom += d;
					denom2 += d * d;
				}

				// (0, v)
				// 縦方向.
				for (int v = 1; v <= r; v++) {
					const auto& p0 = srcSampler(x, y - v);
					const auto& p1 = srcSampler(x, y + v);

					vec3 wr0 = vec3(
						kernelR(abs(p0.r - p.r), sigmaR),
						kernelR(abs(p0.g - p.g), sigmaR),
						kernelR(abs(p0.b - p.b), sigmaR));
					vec3 wr1 = vec3(
						kernelR(abs(p1.r - p.r), sigmaR),
						kernelR(abs(p1.g - p.g), sigmaR),
						kernelR(abs(p1.b - p.b), sigmaR));

					const auto& d0 = depthSampler(x, y - v);
					const auto& d1 = depthSampler(x, y + v);

					const real dd0 = kernelD(d0.w - dc, sigmaD);
					const real dd1 = kernelD(d1.w - dc, sigmaD);

					numer += kernelS(distW, 0, v) * (wr0 * dd0 + wr1 * dd1);
					auto d = kernelS(distW, 0, v) * (wr0 * vec3(p0) * dd0 + wr1 * vec3(p1) * dd1);
					denom += d;
					denom2 += d * d;
				}

				for (int v = 1; v <= r; v++) {
					for (int u = 1; u <= r; u++) {
						const auto& p00 = srcSampler(x - u, y - v);
						const auto& p01 = srcSampler(x - u, y + v);
						const auto& p10 = srcSampler(x + u, y - v);
						const auto& p11 = srcSampler(x + u, y + v);

						vec3 wr00 = vec3(
							kernelR(abs(p00.r - p.r), sigmaR),
							kernelR(abs(p00.g - p.g), sigmaR),
							kernelR(abs(p00.b - p.b), sigmaR));
						vec3 wr01 = vec3(
							kernelR(abs(p01.r - p.r), sigmaR),
							kernelR(abs(p01.g - p.g), sigmaR),
							kernelR(abs(p01.b - p.b), sigmaR));
						vec3 wr10 = vec3(
							kernelR(abs(p10.r - p.r), sigmaR),
							kernelR(abs(p10.g - p.g), sigmaR),
							kernelR(abs(p10.b - p.b), sigmaR));
						vec3 wr11 = vec3(
							kernelR(abs(p11.r - p.r), sigmaR),
							kernelR(abs(p11.g - p.g), sigmaR),
							kernelR(abs(p11.b - p.b), sigmaR));

						const auto& d00 = depthSampler(x - u, y - v);
						const auto& d01 = depthSampler(x - u, y + v);
						const auto& d10 = depthSampler(x + u, y - v);
						const auto& d11 = depthSampler(x + u, y + v);

						const real dd00 = kernelD(d00.w - dc, sigmaD);
						const real dd01 = kernelD(d01.w - dc, sigmaD);
						const real dd10 = kernelD(d10.w - dc, sigmaD);
						const real dd11 = kernelD(d11.w - dc, sigmaD);

						numer += kernelS(distW, u, v) * (wr00 * dd00 + wr01 * dd01 + wr10 * dd10 + wr11 * dd11);
						auto d = kernelS(distW, u, v) * (wr00 * vec3(p00) * dd00 + wr01 * vec3(p01) * dd01 + wr10 * vec3(p10) * dd10 + wr11 * vec3(p11) * dd11);
						denom += d;
						denom2 += d * d;
					}
				}

				auto v = denom / numer;
				dstSampler(x, y) = vec4(v, 1);

				auto v2 = denom2 / (numer * numer);
				v2 -= v * v;
				varSampler.set(x, y, vec4(v2, 1));
			}
		}
	}

	void PracticalNoiseReductionBilateralFilter::operator()(
		const vec4* src,
		const vec4* nml_depth,
		uint32_t width, uint32_t height,
		vec4* dst,
		vec4* variance)
	{
		timer timer;
		timer.begin();

		doBilateralFilter(
			src,
			nml_depth,
			width, height,
			m_sigmaS, m_sigmaR, m_sigmaD,
			dst,
			variance);

		auto elapsed = timer.end();
		AT_PRINTF("Bilateral %f[ms]\n", elapsed);
	}
}
