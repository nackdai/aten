#pragma once

#define NOMINMAX

#include <windows.h>
#include <stdio.h>
#include <utility>

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

	union UnionIdxPtr {
		int idx;
		void* ptr{ nullptr };
	};
}

#define AT_PRINTF		aten::OutputDebugString

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
	#define DEBUG_BREAK()	__debugbreak()

	#ifdef __AT_DEBUG__
		#define AT_ASSERT(b)\
			if (!(b)) {\
				AT_PRINTF("assert : %s(%d)\n", __FILE__, __LINE__);\
				DEBUG_BREAK();\
			}
	#else
		//#define AT_ASSERT(b)
		#define AT_ASSERT(b)\
			if (!(b)) {\
				AT_PRINTF("assert : %s(%d)\n", __FILE__, __LINE__);\
			}
	#endif
#endif

#define AT_VRETURN(b, ret)\
	if (!(b)) {\
		AT_ASSERT(false);\
		return ret;\
	}

#define AT_COUNTOF(a)	(sizeof(a) / sizeof(a[0]))

#define AT_STATICASSERT(b)	static_assert(b, "")

#include "aten_virtual.h"

#ifndef __AT_DEBUG__
#define ENABLE_OMP
#endif

#ifdef ENABLE_OMP
#include <omp.h>
#endif

#ifdef __AT_CUDA__
#include "host_defines.h"
#include "idaten_namespace.h"

#define AT_DEVICE_API		__host__ __device__
#define AT_DEVICE_MTRL_API	__device__
#else
#include "aten_namespace.h"

#define AT_DEVICE_API
#define AT_DEVICE_MTRL_API
#endif
