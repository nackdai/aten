#pragma once

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
}

#define AT_PRINTF		aten::OutputDebugString

#define DEBUG_BREAK()	__debugbreak()

#ifdef __AT_DEBUG__
	#define AT_ASSERT(b)\
        if (!(b)) {\
            AT_PRINTF("assert : %s(%d)\n", __FILE__, __LINE__);\
            DEBUG_BREAK();\
        }
#else
	#define AT_ASSERT(b)
#endif

#define AT_VRETURN(b, ret)\
	if (!(b)) {\
		AT_ASSERT(false);\
		return ret;\
	}

#ifndef __AT_DEBUG__
#define ENABLE_OMP
#endif

#ifdef ENABLE_OMP
#include <omp.h>
#endif
