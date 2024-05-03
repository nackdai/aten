#include "misc/omputil.h"
#include "math/math.h"

namespace aten {
    void OMPUtil::setThreadNum(int32_t num)
    {
#ifdef ENABLE_OMP
        auto maxThreadnum = omp_get_max_threads();
        const auto threadnum = aten::clamp<uint32_t>(num, 1, maxThreadnum);
        omp_set_num_threads(threadnum);
#endif
    }

    int32_t OMPUtil::getThreadNum()
    {
        int32_t ret = 1;
#ifdef ENABLE_OMP
        ret = omp_get_thread_num();
#endif
        return ret;
    }

    int32_t OMPUtil::getThreadIdx()
    {
        int32_t idx = 0;
#ifdef ENABLE_OMP
        idx = omp_get_thread_num();
#endif
        return idx;
    }

    void OMPUtil::InitLock(Lock* lock)
    {
#ifdef ENABLE_OMP
        omp_init_lock(lock);
#endif
    }

    void OMPUtil::SetLock(Lock* lock)
    {
#ifdef ENABLE_OMP
        omp_set_lock(lock);
#endif
    }

    void OMPUtil::UnsetLock(Lock* lock)
    {
#ifdef ENABLE_OMP
        omp_unset_lock(lock);
#endif
    }
}
