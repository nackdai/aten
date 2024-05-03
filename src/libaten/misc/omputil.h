#pragma once

#include "types.h"
#include "defs.h"

namespace aten {
    class OMPUtil {
    private:
        OMPUtil() {}
        ~OMPUtil() {}

    public:
        static void setThreadNum(int32_t num);
        static int32_t getThreadNum();
        static int32_t getThreadIdx();

#ifdef ENABLE_OMP
        using Lock = omp_lock_t;
#else
        using Lock = int32_t;
#endif

        static void InitLock(Lock* lock);
        static void SetLock(Lock* lock);
        static void UnsetLock(Lock* lock);
    };
}
