#include "misc/color.h"

namespace aten {
	const vec3 color::RGB2Y(0.29900, 0.58700, 0.11400);
	const vec3 color::RGB2Cb(-0.16874, -0.33126, 0.50000);
	const vec3 color::RGB2Cr(0.50000, -0.41869, -0.08131);
	const vec3 color::YCbCr2R(1.00000, 0.00000, 1.40200);
	const vec3 color::YCbCr2G(1.00000, -0.34414, -0.71414);
	const vec3 color::YCbCr2B(1.00000, 1.77200, 0.00000);
}