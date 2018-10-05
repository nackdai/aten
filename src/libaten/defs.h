#pragma once

#define NOMINMAX

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <stdio.h>
#include <utility>

#include "types.h"

#define AT_VSPRINTF     vsprintf_s
#define AT_SPRINTF      sprintf_s
#define AT_FPRINTF      fprintf_s

namespace aten {
    inline void OutputDebugString(const char* format, ...)
    {
        va_list argp;
        char buf[256];
        va_start(argp, format);
        AT_VSPRINTF(buf, sizeof(buf), format, argp);
        va_end(argp);

        ::OutputDebugString(buf);
        printf("%s", buf);
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

#include "types.h"

#define AT_VSPRINTF     vsnprintf
#define AT_SPRINTF      snprintf
#define AT_FPRINTF      fprintf

namespace aten {
    using AT_TIME = timeval;
}

#define AT_PRINTF        printf

#define DEBUG_BREAK()    __builtin_trap()
#endif

#ifdef __CUDACC__
    #ifdef __AT_DEBUG__
        #define AT_ASSERT(b)\
            if (!(b)) {\
                printf("assert : %s(%d)\n", __FILE__, __LINE__);\
            }
    #else
        #define AT_ASSERT(b)
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

namespace aten {
    union UnionIdxPtr {
        int idx;
        void* ptr;
    };
}

#include "aten_virtual.h"

#ifndef __AT_DEBUG__
#define ENABLE_OMP
#endif

#ifdef ENABLE_OMP
#include <omp.h>
#endif

#ifdef __AT_CUDA__
#include <host_defines.h>
#include "idaten_namespace.h"

#define AT_DEVICE_API        __host__ __device__
#define AT_DEVICE_MTRL_API    __device__
#else
#include "aten_namespace.h"

#define AT_DEVICE_API
#define AT_DEVICE_MTRL_API
#endif
