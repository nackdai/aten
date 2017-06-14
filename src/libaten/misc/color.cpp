#include "misc/color.h"

namespace AT_NAME {
	const aten::vec3 color::RGB2Y = aten::vec3(real(0.29900), real(0.58700), real(0.11400));
	const aten::vec3 color::RGB2Cb = aten::vec3(real(-0.16874), real(-0.33126), real(0.50000));
	const aten::vec3 color::RGB2Cr = aten::vec3(real(0.50000), real(-0.41869), real(-0.08131));
	const aten::vec3 color::YCbCr2R = aten::vec3(real(1.00000), real(0.00000), real(1.40200));
	const aten::vec3 color::YCbCr2G = aten::vec3(real(1.00000), real(-0.34414), real(-0.71414));
	const aten::vec3 color::YCbCr2B = aten::vec3(real(1.00000), real(1.77200), real(0.00000));
}