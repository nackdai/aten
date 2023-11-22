#pragma once

#include <array>

namespace aten
{
    template<class T> struct array_size;

    template<class T, size_t N>
    struct array_size<std::array<T, N>> {
        static constexpr size_t size = N;
    };
}
