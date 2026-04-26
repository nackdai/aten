#pragma once

#include <array>

#include "atmosphere/sky/sky_constants.h"
#include "atmosphere/sky/unit_quantity.h"

#include "math/math.h"

namespace aten::rainbow {
    // z
    constexpr Length A_MIN = 0.01_mm;
    constexpr Length A_MAX = 0.5_mm;
    constexpr Length A_STEP = 0.02_mm;
    constexpr auto A_WIDTH = static_cast<int32_t>((A_MAX - A_MIN) / A_STEP) + 1;

    // y
    constexpr Length WAVELENGTH_MIN = 360.0_nm;
    constexpr Length WAVELENGTH_MAX = 830.0_nm;
    constexpr Length WAVELENGTH_STEP = 10.0_nm;
    constexpr auto WAVELENGTH_WIDTH = static_cast<int32_t>((WAVELENGTH_MAX - WAVELENGTH_MIN) / WAVELENGTH_STEP) + 1;

    // x
    constexpr auto THETA_MIN = Deg2Rad(36.0F);
    constexpr auto THETA_MAX = Deg2Rad(60.0F);
    constexpr auto THETA_STEP = Deg2Rad(0.1F);
    constexpr auto THETA_WIDTH = static_cast<int32_t>((THETA_MAX - THETA_MIN) / THETA_STEP) + 1;
}
