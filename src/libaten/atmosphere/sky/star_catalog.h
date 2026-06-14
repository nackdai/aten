#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "atmosphere/sky/star_types.h"

namespace aten::sky {
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
