#pragma once

#include "light/ibl.h"
#include "misc/color.h"

// NOTE
// http://www.cs.virginia.edu/~gfx/courses/2007/ImageSynthesis/assignments/envsample.pdf
// http://www.igorsklyar.com/system/documents/papers/4/fiscourse.comp.pdf

namespace aten {
	void ImageBasedLight::preCompute()
	{
		AT_ASSERT(m_envmap);

		auto width = m_envmap->getTexture()->width();
		auto height = m_envmap->getTexture()->height();

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

				auto clr = m_envmap->sample(u, v);
				const auto illum = color::luminance(clr);

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
			for (int i = 0; i < m_cdfV.size(); i++) {
				sum += m_cdfV[i];
				if (i > 0) {
					m_cdfV[i] += m_cdfV[i - 1];
				}
			}
			if (sum > 0) {
				real invSum = 1 / sum;
				for (int i = 0; i < m_cdfV.size(); i++) {
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

	real ImageBasedLight::samplePdf(const ray& r) const
	{
		auto clr = m_envmap->sample(r);
		auto illum = color::luminance(clr);

		auto pdf = illum / m_avgIllum;

		// NOTE
		// 半径１の球の面積で割る.
		pdf /= (4 * AT_MATH_PI);

		return pdf;
	}

	static int samplePdfAndCdf(
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
		int idxTop = 0;
		int idxTail = cdf.size() - 1;

		for (;;) {
			int idxMid = (idxTop + idxTail) >> 1;
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

				int idx = 0;

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
		for (int i = 0; i < cdf.size(); i++) {
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

	LightSampleResult ImageBasedLight::sample(const vec3& org, sampler* sampler) const
	{
		const auto r1 = sampler->nextSample();
		const auto r2 = sampler->nextSample();

		real pdfU, pdfV;
		real cdfU, cdfV;

		int y = samplePdfAndCdf(r1, m_cdfV, pdfV, cdfV);
		int x = samplePdfAndCdf(r2, m_cdfU[y], pdfU, cdfU);

		auto width = m_envmap->getTexture()->width();
		auto height = m_envmap->getTexture()->height();

		real u = (real)(x + 0.5) / width;
		real v = (real)(y + 0.5) / height;

		LightSampleResult result;

		// NOTE
		// p(w) = p(u, v) * (w * h) / (2π^2 * sin(θ))
		auto pi2 = AT_MATH_PI * AT_MATH_PI;
		auto theta = AT_MATH_PI * v;
		result.pdf = (pdfU * pdfV) * ((width * height) / (pi2 * aten::sin(theta)));

		// u, v -> direction.
		result.dir = envmap::convertUVToDirection(u, v);

		result.le = m_envmap->sample(u, v);
		result.intensity = real(1);
		result.finalColor = result.le * result.intensity;

		// TODO
		// Currently not used...
		result.pos = vec3();
		result.nml = vec3();

		return std::move(result);
	}
}