#include <vector>
#include "denoise/PracticalNoiseReduction.h"
#include "misc/color.h"

namespace aten {
	void PracticalNoiseReduction::operator()(
		const vec4* src,
		uint32_t width, uint32_t height,
		vec4* dst)
	{
		std::vector<vec4> filtered(width * height);
		std::vector<vec4> var_filtered(width * height);
		std::vector<vec4> hv(width * height);

		const real v_p = 8;
		const real v_c = 0.01;
		const real v_d = 2;

		const real t = 0.002;

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
				for (int yy = -1; yy <= 1; yy++) {
					for (int xx = -1; xx <= 1; xx++) {
						int p_x = aten::clamp<int>(x + xx, 0, width - 1);
						int p_y = aten::clamp<int>(y + yy, 0, height - 1);

						if (p_x == x && p_y == y) {
							continue;
						}

						int p = p_y * width + p_x;

						const vec4& ci = m_indirect[p];
						real di = m_nml_depth[p].w;

						real l_p = (vec3(p_x, p_y, 0) - p0).length();
						real l_c = (ci - c0).length();
						real l_d = aten::abs(di - d0);

						real g_p = aten::exp(-0.5 * l_p * l_p / (v_p * v_p));
						real g_c = aten::exp(-0.5 * l_c * l_c / (v_c * v_c));
						real g_d = aten::exp(-0.5 * l_d * l_d / (v_d * v_d));

						real weight = g_p + g_c + g_d;

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

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int pos = y * width + x;
				
				const real Lf = color::RGBtoY(filtered[pos]);
				const real Llv = color::RGBtoY(m_direct[pos]);

				const real Lb = Lf + Llv;
				real Lb2 = Lb * Lb;

				const real varLu = color::RGBtoY(m_variance[pos]);
				const real varLf = color::RGBtoY(var_filtered[pos]);

				const real u = varLu / Lb2;
				const real f = varLf / Lb2;

				const real D = t * u + t * f - u * f;

				real s = 0;

				if (D < 0) {
					s = 0;
				}
				else if (u <= t) {
					s = 1;
				}
				else {
					s = (f + aten::sqrt(D)) / (u + f);
				}

				hv[pos] = s * m_indirect[pos] + (1 - s) * filtered[pos];
			}
		}

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int pos = y * width + x;

				dst[pos] = m_direct[pos] + hv[pos];
			}
		}
	}
}
