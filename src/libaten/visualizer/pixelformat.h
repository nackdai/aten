#pragma once

namespace aten {
    enum class PixelFormat : int32_t {
        rgba8,
        rgba32f,
        rgba16f,
    };

    inline int32_t GetBytesPerPxiel(PixelFormat fmt)
    {
        constexpr int32_t bpp[] = {
            4, 16, 8,
        };
        return bpp[static_cast<size_t>(fmt)];
    }
}
