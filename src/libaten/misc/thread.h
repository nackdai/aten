#pragma once

#include "types.h"
#include "defs.h"

namespace aten {
	class thread {
	private:
		thread() {}
		~thread() {}

	public:
		static void setThreadNum(uint32_t num);

		static uint32_t getThreadNum()
		{
			return g_threadnum;
		}

		static int getThreadIdx();

	private:
		static uint32_t g_threadnum;
	};
}