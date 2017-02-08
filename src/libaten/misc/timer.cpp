#include "defs.h"
#include "misc/timer.h"

namespace aten {
	static int64_t s_freq = 0;

	void timer::init()
	{
		if (s_freq == 0) {
			LARGE_INTEGER freq;
			memset(&freq, 0, sizeof(freq));

			auto result = QueryPerformanceFrequency(&freq);
			AT_ASSERT(result);

			s_freq = freq.QuadPart;
		}
	}

	void timer::begin()
	{
		LARGE_INTEGER b;
		memset(&b, 0, sizeof(b));

		auto result = QueryPerformanceCounter(&b);
		AT_ASSERT(result);

		m_begin = b.QuadPart;
	}

	real timer::end()
	{
		LARGE_INTEGER cur;
		QueryPerformanceCounter(&cur);

		auto time = (cur.QuadPart - m_begin) * 1000.0f / s_freq;
		return time;
	}
}