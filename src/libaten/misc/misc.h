#pragma once

#include <array>

namespace aten
{
    template<typename T> struct array_size;

    template<typename T, size_t N>
    struct array_size<std::array<T, N>> {
        static constexpr size_t size = N;
    };
}
