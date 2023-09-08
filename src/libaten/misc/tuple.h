#pragma once

#ifdef __CUDACC__
#include <cuda/std/tuple>
#else
#include <tuple>
#endif

// NOTE
// https://stackoverflow.com/questions/41660311/aliasing-a-std-template-function
// https://stackoverflow.com/questions/2821223/how-would-one-call-stdforward-on-all-arguments-in-a-variadic-function

namespace aten {
#ifdef __CUDACC__
    template <typename... _Types>
    using tuple = cuda::std::tuple<_Types...>;

    template <typename... _Types>
    __host__ __device__ decltype(auto) make_tuple(_Types... args) {
        return cuda::std::make_tuple<_Types...>(std::forward<_Types>(args)...);
    }

    template <std::size_t N, typename _Type>
    constexpr __host__ __device__ decltype(auto) get(_Type&& arg) {
        return cuda::std::get<N>(std::forward<decltype(arg)>(arg));
    }

    template <typename... Args>
    constexpr __host__ __device__ decltype(auto) tie(Args&&... args) {
        return cuda::std::tie(std::forward<Args>(args)...);
    };
#else
    template <typename... _Types>
    using tuple = std::tuple<_Types...>;

    template <typename... _Types>
    decltype(auto) make_tuple(_Types... args) {
        return std::make_tuple<_Types...>(std::forward<_Types>(args)...);
    }

    template <std::size_t N, typename _Type>
    constexpr decltype(auto) get(_Type&& arg) {
        return std::get<N>(std::forward<decltype(arg)>(arg));
    }

    template <typename... Args>
    constexpr decltype(auto) tie(Args&&... args) {
        return std::tie(std::forward<Args>(args)...);
    };
#endif
}
