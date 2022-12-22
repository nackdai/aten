#pragma once

#ifdef __CUDACC__
#include <thrust/tuple.h>
#else
#include <tuple>
#endif

// NOTE
// https://stackoverflow.com/questions/41660311/aliasing-a-std-template-function
// https://stackoverflow.com/questions/2821223/how-would-one-call-stdforward-on-all-arguments-in-a-variadic-function

namespace aten {
#ifdef __CUDACC__
    template <class... _Types>
    using tuple = thrust::tuple<_Types...>;

    template <class... _Types>
    __host__ __device__ decltype(auto) make_tuple(_Types... args) {
        return thrust::make_tuple<_Types...>(std::forward<_Types>(args)...);
    }

    template <std::size_t N, class _Type>
    __host__ __device__ decltype(auto) get(_Type&& arg) {
        return thrust::get<N>(std::forward<decltype(arg)>(arg));
    }
#else
    template <class... _Types>
    using tuple = std::tuple<_Types...>;

    template <class... _Types>
    decltype(auto) make_tuple(_Types... args) {
        return std::make_tuple<_Types...>(std::forward<_Types>(args)...);
    }

    template <std::size_t N>
    auto get = [](auto&& arg) -> decltype(auto) {
        return std::get<N>(std::forward<decltype(arg)>(arg));
    };
#endif
}
