#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "math/vec3.h"

namespace aten::sky {
    // Normalized star data used by the renderer. Both the ASCII and binary BSC5
    // loaders convert their source-specific fields into this common form.
    struct Star {
        // J2000 equatorial direction. This is not yet transformed to the local
        // horizon frame; that should be done later from observation time/location.
        aten::vec3 direction_j2000{ 0.0F, 1.0F, 0.0F };

        // Irradiance derived from visual_magnitude using the night-sky paper's
        // star magnitude equation.
        float irradiance{ 0.0F };

        // Linear RGB chromaticity-like color normalized to max component 1.
        // Brightness is carried separately by irradiance.
        aten::vec3 color{ 1.0F };

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

    class StarCatalog {
    public:
        // Backward-compatible default: load the fixed-width ASCII BSC catalog.
        bool LoadFromBrightStarCatalog(const std::string& path);

        // Load the CDS/VizieR-style fixed-width ASCII catalog file.
        bool LoadFromBrightStarCatalogAscii(const std::string& path);

        // Load the common BSC5 binary file with a 7-int header and 32-byte records.
        bool LoadFromBrightStarCatalogBinary(const std::string& path);

        const std::vector<Star>& stars() const
        {
            return stars_;
        }

        void Clear()
        {
            stars_.clear();
        }

    private:
        std::vector<Star> stars_;
    };
}
