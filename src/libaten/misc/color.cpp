#include "misc/color.h"

namespace aten {
	const vec3 color::RGB2Y(real(0.29900), real(0.58700), real(0.11400));
	const vec3 color::RGB2Cb(real(-0.16874), real(-0.33126), real(0.50000));
	const vec3 color::RGB2Cr(real(0.50000), real(-0.41869), real(-0.08131));
	const vec3 color::YCbCr2R(real(1.00000), real(0.00000), real(1.40200));
	const vec3 color::YCbCr2G(real(1.00000), real(-0.34414), real(-0.71414));
	const vec3 color::YCbCr2B(real(1.00000), real(1.77200), real(0.00000));
}