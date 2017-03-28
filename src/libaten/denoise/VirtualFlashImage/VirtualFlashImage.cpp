#include <vector>
#include "denoise/VirtualFlashImage/VirtualFlashImage.h"

// filtering parameter for each step!
#define FILTER_SIZE_STEP1	7
#define FILTER_SIZE_STEP2	31
#define STD_D_STEP1		real(1.0)
#define STD_D_STEP2		real(5.0)

#define MAX_DF	101
#define CONFIDENCE_LEVEL1	5
#define CONFIDENCE_LEVEL2	4

#define MAX_HALF_PATCH_SIZE 2	// patch size

/////////////////////////////////////////////////////
// t-distribution value.
// [0: 80%, 1: 90%, 2: 95%, 3: 98%, 4: 99%, 5: 99.8%]
// 信頼区間を計算するためのt-分布のテーブル.
static const aten::real ttable[][6] = {
#include "denoise/VirtualFlashImage/t_table.dat"
};

namespace aten {
	static void gaussianFilter(
		int cx, int cy,
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

	static inline bool isInCI(const vec4& p, const vec4& low, const vec4& high)
	{
		return (((low.x < p.x) && (high.x > p.x)) &&
			((low.y < p.y) && (high.y > p.y)) &&
			((low.z < p.z) && (high.z > p.z)));
	}

	static real euclideanDist(const vec4& a, const vec4& b)
	{
		return ((b.x - a.x) * (b.x - a.x) +
			(b.y - a.y) * (b.y - a.y) +
			(b.z - a.z) * (b.z - a.z));
	}

#define CI_FOR_FLASH
#define ADAPTIVE_PATCH

	void filterStep1(
		int cx, int cy,
		const vec4* _in,
		const vec4* _flash,
		const vec4* _std,		// 標準偏差.
		const vec4* _stdFlash,	// Flash画像の標準偏差.
		vec4* _out, 
		int width, int height, 
		int filter_size, real std_d,
		int numSamples, 
		const real* _tvalue, 
		vec4* _stdresult)
	{
		int halfWindowSize = filter_size / 2;
		int _df = std::min(numSamples - 1, MAX_DF);	// 自由度.

		const real RANGE_CONSTANCE = real(2.0);

		real arrWeights[49];
		int arrIdx[49];

		// 信頼区間の基準値.
		vec4 CI;

		// ガウスフィルタ係数.
		// g(x) = exp(-1/2 * x^2/d^2) = exp(-(x * x) / (2 * d * d))
		real std_d2 = 2 * std_d * std_d;

		const auto cIdx = cy * width + cx;

		const auto& sf = _stdFlash[cIdx];
		const auto& std = _std[cIdx];

		auto tmp_adaptiveRange = RANGE_CONSTANCE * aten::sqrt(aten::abs(0.99 * sf * sf + 0.01 * std * std));
		auto adaptiveRange2 = 2 * tmp_adaptiveRange * tmp_adaptiveRange;

		auto adaptiveRange = (adaptiveRange2.x + adaptiveRange2.y + adaptiveRange2.z) / 3;
		adaptiveRange = std::max(adaptiveRange, AT_MATH_EPSILON);

		const auto& curFlash = _flash[cIdx];

		// Flash画像の中心の信頼区間を計算.
#ifdef CI_FOR_FLASH
		CI = _tvalue[_df] * sf;
		const vec4 CI_center_low = curFlash - CI - AT_MATH_EPSILON;
		const vec4 CI_center_high = curFlash + CI + AT_MATH_EPSILON;
#endif

#ifdef ADAPTIVE_PATCH	
		real h = adaptiveRange * 2048;
		int halfPatchSize = std::min<int>(h, MAX_HALF_PATCH_SIZE);
#else	
		int halfPatchSize = MAX_HALF_PATCH_SIZE;
#endif

		const auto& curImg = _in[cIdx];

		// 計算範囲ウインドウ開始位置.
		int startWindow_x = std::max(0, cx - halfWindowSize);
		int startWindow_y = std::max(0, cy - halfWindowSize);

		// 計算範囲ウインドウ終了位置.
		int endWindow_x = std::min(width - 1, cx + halfWindowSize);
		int endWindow_y = std::min(height - 1, cy + halfWindowSize);

		// 中心の分散
		const auto cvar = _std[cIdx] * _std[cIdx];

		int numNeighboors = 0;
		real sumWeight = 0;

		for (int iy = startWindow_y; iy <= endWindow_y; ++iy) {
			for (int ix = startWindow_x; ix <= endWindow_x; ++ix) {
				const auto idx = iy * width + ix;

				const auto& tarImg = _in[idx];

				// 分散.
				const auto var = _std[idx] * _std[idx];

				// 自由度.
				int df[4];
				{
					const vec4 denominator = (cvar * cvar + var * var) / (real)(numSamples - 1);

					for (int c = 0; c < 4; c++) {
						if (denominator[c] < AT_MATH_EPSILON) {
							df[c] = MAX_DF;
						}
						else {
							df[c] = (int)((cvar[c] + var[c]) * (cvar[c] + var[c]) / denominator[c] + 0.5);
						}

						df[c] = std::min<int>(df[c], MAX_DF);
					}
				}

				// 信頼区間を計算.
				CI = vec4(
					_tvalue[df[0]] * aten::sqrt(aten::abs(cvar[0] + var[0])),
					_tvalue[df[1]] * aten::sqrt(aten::abs(cvar[1] + var[1])),
					_tvalue[df[2]] * aten::sqrt(aten::abs(cvar[2] + var[2])),
					0);
				const vec4 CI_max = tarImg - curImg + CI + AT_MATH_EPSILON;
				const vec4 CI_min = tarImg - curImg - CI - AT_MATH_EPSILON;

				// Check if CI_min < 0 && CI_max > 0.
				if (!isInCI(vec4(0), CI_min, CI_max)) {
					continue;
				}

				arrIdx[numNeighboors] = idx;

				// ピクセル距離をガウスフィルタ.
				const auto imageDist = (real)(cx - ix) * (cx - ix) + (cy - iy) * (cy - iy);
				real weight = aten::exp(-imageDist / std_d2);

#ifdef CI_FOR_FLASH
				const auto& tarColor = _flash[idx];
				CI = _tvalue[_df] * vec4(_stdFlash[idx].x, _stdFlash[idx].y, _stdFlash[idx].z, 0);
				auto CI_target_low = tarColor - CI - AT_MATH_EPSILON;
				auto CI_target_high = tarColor + CI + AT_MATH_EPSILON;
#endif

				real countPass = 0;
				real colorDist = 0;

				// Flash画像の色距離についてNonLocalMeanフィルタ?
				for (int yy = -halfPatchSize; yy <= halfPatchSize; ++yy) {
					for (int xx = -halfPatchSize; xx <= halfPatchSize; ++xx) {
						int ccx = aten::clamp(cx + xx, 0, width - 1);
						int ccy = aten::clamp(cy + yy, 0, height - 1);

						int iix = aten::clamp(ix + xx, 0, width - 1);
						int iiy = aten::clamp(iy + yy, 0, height - 1);

						int cpos = ccy * width + ccx;
						int ipos = iiy * width + iix;

						const auto& curPix = _flash[cpos];
						const auto& tarPix = _flash[ipos];

#ifdef CI_FOR_FLASH
						if (isInCI(curPix, CI_center_low, CI_center_high) && isInCI(tarPix, CI_target_low, CI_target_high))
#endif
						{
							colorDist += euclideanDist(curPix, tarPix);
							countPass += real(3);
						}
					}
				}

				weight *= aten::exp(-colorDist / (adaptiveRange * countPass));

				arrWeights[numNeighboors] = weight;
				++numNeighboors;

				_out[cIdx] += weight * _in[idx];
				sumWeight += weight;
			}
		}

#if 1
		if (_stdresult) {
			vec4 reconVar(0);
			for (int ii = 0; ii < numNeighboors; ++ii) {
				reconVar += arrWeights[ii] * arrWeights[ii] * _std[arrIdx[ii]] * _std[arrIdx[ii]];
			}

			static const real COVARIANCE = real(1);
			for (int ii = 0; ii < numNeighboors; ++ii) {
				for (int jj = ii + 1; jj < numNeighboors; ++jj) {
					reconVar += arrWeights[ii] * arrWeights[jj]
						* COVARIANCE
						* aten::sqrt(aten::abs(_std[arrIdx[ii]] * _std[arrIdx[ii]] * _std[arrIdx[jj]] * _std[arrIdx[jj]]));
				}
			}

			_stdresult[cIdx] = aten::sqrt(aten::abs(reconVar / (sumWeight * sumWeight)));
		}
#endif
		_out[cIdx] /= sumWeight;
	}

	void filterStep2(
		int cx, int cy,
		const vec4* _image,
		const vec4* _flash,
		const vec4* _std, 
		const vec4* _stdFlash, 
		vec4* _out, 
		int width, int height, 
		int filter_size, real std_d, 
		int numSamples, 
		const real* _tvalue)
	{
		const int halfWindowSize = filter_size / 2;
		const int _df = std::min<int>(numSamples - 1, MAX_DF);	// 自由度.

		const real RANGE_CONSTANCE = real(2);

		const real std_d2 = 2.0f * std_d * std_d;

		// 信頼区間.
		vec4 CI;

		const int cIdx = cy * width + cx;

		const auto& sf = _stdFlash[cIdx];
		const auto& std = _std[cIdx];

		auto tmp_adaptiveRange = RANGE_CONSTANCE * aten::sqrt(aten::abs(0.99f * sf * sf + 0.01f * std * std));
		const auto adaptiveRange2 = 2 * tmp_adaptiveRange * tmp_adaptiveRange;

		auto adaptiveRange = (adaptiveRange2[0] + adaptiveRange2[1] + adaptiveRange2[2]) / 3.0f;
		adaptiveRange = std::max<real>(adaptiveRange, AT_MATH_EPSILON);

		const auto tvalue = _tvalue[_df];

		// Flash画像の中心の信頼区間を計算.
#ifdef CI_FOR_FLASH
		const auto& curColor = _flash[cIdx];
		CI = tvalue * vec4(sf.x, sf.y, sf.z, 0);
		const auto CI_center_low = curColor - CI - AT_MATH_EPSILON;
		const auto CI_center_high = curColor + CI + AT_MATH_EPSILON;
#endif

		const auto& curImg = _image[cIdx];

		// 入力画像の中心の信頼区間を計算.
		CI = tvalue * vec4(std.x, std.y, std.z, 0);
		const auto CI_min = curImg - CI - AT_MATH_EPSILON;
		const auto CI_max = curImg + CI + AT_MATH_EPSILON;

#ifdef ADAPTIVE_PATCH	
		real h = adaptiveRange * 2048.0f;
		int halfPatchSize = std::min((int)h, MAX_HALF_PATCH_SIZE);
#else	
		int halfPatchSize = MAX_HALF_PATCH_SIZE;
#endif

		// 計算範囲ウインドウ開始位置.
		int startWindow_x = std::max(0, cx - halfWindowSize);
		int startWindow_y = std::max(0, cy - halfWindowSize);

		// 計算範囲ウインドウ終了位置.
		int endWindow_x = std::min(width - 1, cx + halfWindowSize);
		int endWindow_y = std::min(height - 1, cy + halfWindowSize);

		real sumWeight = 0;

		for (int iy = startWindow_y; iy <= endWindow_y; ++iy) {
			for (int ix = startWindow_x; ix <= endWindow_x; ++ix) {
				int idx = iy * width + ix;
				const auto& tarImg = _image[idx];

				// Check if in CI.
				if (!isInCI(tarImg, CI_min, CI_max)) {
					continue;
				}

				const real imageDist = (real)(cx - ix) * (cx - ix) + (cy - iy) * (cy - iy);
				real weight = aten::exp(-imageDist / std_d2);

#ifdef CI_FOR_FLASH
				const auto& tarColor = _flash[idx];
				CI = tvalue * vec4(_stdFlash[idx].x, _stdFlash[idx].y, _stdFlash[idx].z, 0);
				const auto CI_target_low = tarColor - CI - AT_MATH_EPSILON;
				const auto CI_target_high = tarColor + CI + AT_MATH_EPSILON;
#endif

				real countPass = 0;
				real colorDist = 0;

				for (int yy = -halfPatchSize; yy <= halfPatchSize; ++yy) {
					for (int xx = -halfPatchSize; xx <= halfPatchSize; ++xx) {
						int ccx = aten::clamp(cx + xx, 0, width - 1);
						int ccy = aten::clamp(cy + yy, 0, height - 1);

						int iix = aten::clamp(ix + xx, 0, width - 1);
						int iiy = aten::clamp(iy + yy, 0, height - 1);

						int cpos = ccy * width + ccx;
						int ipos = iiy * width + iix;

						const auto& curPix = _flash[cpos];
						const auto& tarPix = _flash[ipos];

#ifdef CI_FOR_FLASH
						if (isInCI(curPix, CI_center_low, CI_center_high) && isInCI(tarPix, CI_target_low, CI_target_high))
#endif
						{
							colorDist += euclideanDist(curPix, tarPix);
							countPass += real(3);
						}
					}
				}

				weight *= aten::exp(-colorDist / (adaptiveRange * countPass));

				_out[cIdx] += weight * tarImg;
				sumWeight += weight;
			}
		}

		_out[cIdx] /= sumWeight;
	}

	void VirtualFlashImage::operator()(
		const vec4* src,
		uint32_t width, uint32_t height,
		vec4* dst)
	{
		AT_ASSERT(m_numSamples > 0);
		AT_ASSERT(m_variance);
		AT_ASSERT(m_flash);
		AT_ASSERT(m_varFlash);

		std::vector<vec4> stdSrc(width * height);
		std::vector<vec4> stdFlash(width * height);

		// 標準偏差を計算.
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				auto pos = y * width + x;

				stdSrc[pos] = aten::sqrt(aten::abs(m_variance[pos]));
				stdFlash[pos] = aten::sqrt(aten::abs(m_varFlash[pos]));
			}
		}

		// 標準偏差にガウスフィルタをかける.
		{
			std::vector<vec4> tmp(width * height);

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
			for (int cy = 0; cy < height; cy++) {
				for (int cx = 0; cx < width; cx++) {
					gaussianFilter(
						cx, cy,
						&stdFlash[0],
						&tmp[0],
						width, height,
						FILTER_SIZE_STEP1, STD_D_STEP1);
				}
			}

			memcpy(&stdFlash[0], &tmp[0], width * height * sizeof(vec4));
		}

		const auto numSamples = m_numSamples;

		int df = std::min<uint32_t>(numSamples - 1, MAX_DF);	// 自由度.

		std::vector<real> tvalue(MAX_DF + 1);
		std::vector<real> tvalue2(MAX_DF + 1);
		for (int i = 0; i < MAX_DF + 1; i++) {
			tvalue[i] = ttable[i][CONFIDENCE_LEVEL1];
			tvalue2[i] = ttable[i][CONFIDENCE_LEVEL2];
		}

		std::vector<vec4> tmpStd(width * height);
		std::vector<vec4> tmpDenoised(width * height);

#if 1
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int cy = 0; cy < height; cy++) {
			for (int cx = 0; cx < width; cx++) {
				filterStep1(
					cx, cy,
					src,
					m_flash,
					&stdSrc[0],
					&stdFlash[0],
					&tmpDenoised[0],
					width, height,
					FILTER_SIZE_STEP1, STD_D_STEP1,
					numSamples,
					&tvalue[0],
					&tmpStd[0]);
			}
		}
#endif

#if 1
		memcpy(&stdSrc[0], &tmpStd[0], width * height * sizeof(vec4));

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int cy = 0; cy < height; cy++) {
			for (int cx = 0; cx < width; cx++) {
				filterStep2(
					cx, cy,
					&tmpDenoised[0],
					m_flash,
					&stdSrc[0],
					&stdFlash[0],
					dst,
					width, height,
					FILTER_SIZE_STEP2, STD_D_STEP2,
					numSamples,
					&tvalue2[0]);
			}
		}
#endif
	}
}
