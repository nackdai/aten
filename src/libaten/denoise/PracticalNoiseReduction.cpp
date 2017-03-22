#include <vector>
#include "denoise/PracticalNoiseReduction.h"
#include "misc/color.h"
#include "denoise/bilateral.h"

#pragma optimize( "", off )  

namespace aten {
	void PracticalNoiseReduction::operator()(
		const vec4* src,
		uint32_t width, uint32_t height,
		vec4* dst)
	{
		AT_PRINTF("PracticalNoiseReduction\n");

		std::vector<vec4> filtered(width * height);
		std::vector<vec4> var_filtered(width * height);
		std::vector<vec4> hv(width * height);

		const real v_p = 5;
		const real v_c = 0.01;
		const real v_d = 2;

		const real t = 0.1;

#if 0
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				real sumW = 0;

				int pos = y * width + x;

				const vec3 p0(x, y, 0);
				const vec4& c0 = m_indirect[pos];
				real d0 = m_nml_depth[pos].w;

				vec4 color;
				vec4 col2;

				// 3x3.
				static const int Radius = 3;
				static const int HalfRadius = Radius / 2;

				for (int yy = -HalfRadius; yy <= HalfRadius; yy++) {
					for (int xx = -HalfRadius; xx <= HalfRadius; xx++) {
						int p_x = aten::clamp<int>(x + xx, 0, width - 1);
						int p_y = aten::clamp<int>(y + yy, 0, height - 1);

						int p = p_y * width + p_x;

						const vec4& ci = m_indirect[p];
						real di = m_nml_depth[p].w;

						real l_p = (vec3(p_x, p_y, 0) - p0).length();
						real l_c = (ci - c0).length();
						real l_d = aten::abs(di - d0);

						real g_p = 1 / aten::sqrt(2 * AT_MATH_PI * v_p) * aten::exp(-0.5 * l_p * l_p / (v_p * v_p));
						real g_c = 1 / aten::sqrt(2 * AT_MATH_PI * v_c) * aten::exp(-0.5 * l_c * l_c / (v_c * v_c));
						real g_d = 1 / aten::sqrt(2 * AT_MATH_PI * v_d) * aten::exp(-0.5 * l_d * l_d / (v_d * v_d));

						real weight = g_p * g_c * g_d;

						color += weight * ci;
						col2 += color * color;
						sumW += weight;
					}
				}

				color /= sumW;

				filtered[pos] = color;

				col2 /= (sumW * sumW);
				var_filtered[pos] = col2 - color * color;
			}
		}
#else
		BilateralFilter filter;
		filter.setVarianceBuffer(&var_filtered[0]);
		filter(m_indirect, width, height, &filtered[0]);
#endif

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int pos = y * width + x;
				
				const vec4& Lf = filtered[pos];
				const vec4& Llv = m_direct[pos];

				const vec4 Lb = Lf + Llv;
				const vec4 Lb2 = Lb * Lb + vec4(0.0001);

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

				hv[pos] = s * m_indirect[pos] + (vec4(1) - s) * filtered[pos];
			}
		}

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int pos = y * width + x;

				//dst[pos] = m_indirect[pos];
				dst[pos] = filtered[pos];
				//dst[pos] = m_direct[pos] + hv[pos];
			}
		}
	}
}
