#include "misc/omputil.h"
#include "math/math.h"

namespace aten {
	uint32_t OMPUtil::g_threadnum = 1;
	
	void OMPUtil::setThreadNum(uint32_t num)
	{
#ifdef ENABLE_OMP
		auto maxThreadnum = omp_get_max_threads();
#else
		auto maxThreadnum = 8;
#endif

		g_threadnum = aten::clamp<uint32_t>(num, 1, maxThreadnum);

#ifdef ENABLE_OMP
		omp_set_num_threads(g_threadnum);
#endif
	}

	int OMPUtil::getThreadIdx()
	{
		int idx = 0;
#ifdef ENABLE_OMP
		idx = omp_get_thread_num();
#endif
		return idx;
	}
}