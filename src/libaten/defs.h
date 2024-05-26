#pragma once

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#define NOMINMAX

// NOTE:
// In C++20, we can use std::format.
namespace _format {
    namespace detail {
        // https://an-embedded-engineer.hateblo.jp/entry/2020/08/23/161317

        // To call std::string::c_str automatically.
        template<class T>
        inline auto Convert(T&& value)
        {
            if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::string>) {
                return std::forward<T>(value).c_str();
            }
            else {
                return std::forward<T>(value);
            }
        }

        template<class ... Args>
        std::string StringFormat(const std::string_view format, Args&& ... args)
        {
            // NOTE:
            // https://en.cppreference.com/w/cpp/io/c/fprintf
            // Calling std::snprintf with zero buf_size and null pointer for buffer is useful (when the overhead of double-call is acceptable) to determine the necessary buffer size to contain the output:
            const auto str_len = std::snprintf(nullptr, 0, format.data(), std::forward<Args>(args) ...);

            if (str_len < 0) {
                throw std::runtime_error("String Formatting Error");
            }

            // Add 1 byte for null character.
            const auto buffer_size = str_len + sizeof(char);

            std::unique_ptr<char[]> buffer(new char[buffer_size]);

            std::snprintf(buffer.get(), buffer_size, format.data(), args ...);

            return std::string(buffer.get(), buffer.get() + str_len);
        }
    }
}

namespace aten {
    template<typename ... Args>
    inline std::string StringFormat(const std::string_view format, Args&& ... args)
    {
        return _format::detail::StringFormat(
            format,
            _format::detail::Convert(std::forward<Args>(args)) ...);
    }
}

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <stdio.h>
#include <utility>
#include <cstring>

#include "types.h"

namespace aten {
    template<typename ... Args>
    inline void OutputDebugString(const std::string_view format, Args&& ... args)
    {
        auto display_str = aten::StringFormat(format, std::forward<Args>(args) ...);
        ::OutputDebugString(display_str.c_str());
        std::cout << display_str;
    }

    using AT_TIME = int64_t;
}

#define AT_PRINTF        aten::OutputDebugString

#define DEBUG_BREAK()    __debugbreak()
#else
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <utility>
#include <cstring>

#include "types.h"

namespace aten {
    using AT_TIME = timeval;
}

#define AT_PRINTF        printf

#define DEBUG_BREAK()    __builtin_trap()
#endif

#ifdef __AT_CUDA__
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "idaten_namespace.h"

#define AT_HOST_DEVICE_API  __host__ __device__
#define AT_DEVICE_API       __device__
#else
#include "aten_namespace.h"

#define AT_HOST_DEVICE_API
#define AT_DEVICE_API
#endif

#ifdef __CUDACC__
#define AT_INLINE __forceinline__
#else
#define AT_INLINE inline
#endif

#ifdef __AT_DEBUG__
#define AT_INLINE_RELEASE
#else
#define AT_INLINE_RELEASE   AT_INLINE
#endif


#include <assert.h>

#ifdef __CUDACC__
    #ifdef __AT_DEBUG__
        #define AT_ASSERT(b)\
            if (!(b)) {\
                printf("assert : %s(%d)\n", __FILE__, __LINE__);\
                assert(false);\
            }
    #else
        #define AT_ASSERT(b)    assert(false);
    #endif
#else
    #ifdef __AT_DEBUG__
        #define AT_ASSERT(b)\
            if (!(b)) {\
                AT_PRINTF("assert : %s(%d)\n", __FILE__, __LINE__);\
                DEBUG_BREAK();\
            }
    #else
        #define AT_ASSERT(b)\
            if (!(b)) {\
                AT_PRINTF("assert : %s(%d)\n", __FILE__, __LINE__);\
            }
    #endif
#endif

#ifdef __AT_DEBUG__
    #define AT_ASSERT_LOG(b, log)\
        if (!(b)) {\
            AT_PRINTF("assert : %s(%d)\n", __FILE__, __LINE__);\
            AT_PRINTF("    [%s]\n", log);\
            DEBUG_BREAK();\
        }
#else
    #define AT_ASSERT_LOG(b, log)\
        if (!(b)) {\
            AT_PRINTF("assert : %s(%d)\n", __FILE__, __LINE__);\
            AT_PRINTF("    [%s]\n", log);\
        }
#endif


#define AT_VRETURN(b, ret)\
    if (!(b)) {\
        AT_ASSERT(false);\
        return ret;\
    }

#define AT_VRETURN_FALSE(b)    AT_VRETURN(b, false)

#define AT_COUNTOF(a)    (sizeof(a) / sizeof(a[0]))

#define AT_STATICASSERT(b)    static_assert(b, "")

#ifndef __AT_DEBUG__
#define ENABLE_OMP
#endif

#ifdef ENABLE_OMP
#include <omp.h>
#endif
