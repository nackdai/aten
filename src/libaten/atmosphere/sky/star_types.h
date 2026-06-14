#pragma once

#include <cstdint>

#include "math/vec3.h"

namespace aten::sky {
    // Renderer-ready star data. The catalog loader converts source-specific
    // fields into this common layout while loading, so this array can be used
    // directly as the CPU/GPU render input without another packing step.
    struct Star {
        // J2000 equatorial direction. This is not yet transformed to the local
        // horizon frame; that should be done later from observation time/location.
        aten::vec3 direction_j2000{ 0.0F, 1.0F, 0.0F };

        // Irradiance derived from visual_magnitude using the night-sky paper's
        // star magnitude equation.
        float irradiance{ 0.0F };

        // Linear RGB weight used to distribute the scalar Vmag-derived irradiance
        // into RGB. It is derived from B-V/Teff blackbody color and normalized so
        // its linear-sRGB luminance is 1.
        aten::vec3 rgb_irradiance_weight{ 1.0F };

        // V-band apparent magnitude. Smaller values are brighter.
        float visual_magnitude{ 0.0F };

        // B-V color index. ASCII BSC provides this directly; binary BSC5 does not,
        // so the binary loader estimates it from the spectral-type shorthand.
        float bv_color{ 0.0F };

        // Proper motion in arcsec/year. For ASCII BSC, pmRA follows the catalog
        // convention cos(Dec) * dRA/dt. The binary loader converts rad/year to
        // arcsec/year for this common representation.
        float proper_motion_ra{ 0.0F };
        float proper_motion_dec{ 0.0F };

        // Harvard Revised Number, used as the stable Bright Star Catalog id.
        uint32_t hr{ 0 };
    };
}
