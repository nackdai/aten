#pragma once

#include <array>
#include <utility>

namespace aten
{
    template<class T> struct array_size;

    template<class T, size_t N>
    struct array_size<std::array<T, N>> {
        static constexpr size_t size = N;
    };

    template <class T>
    inline AT_DEVICE_API auto swap(T& a, T& b) -> std::enable_if_t<!std::is_const_v<T>>
    {
#ifdef __CUDACC__
        const auto t = a;
        a = b;
        b = t;
#else
        std::swap(a, b);
#endif
    }
}
