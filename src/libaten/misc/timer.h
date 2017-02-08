#pragma once

#include "types.h"

namespace aten {
	class timer {
	public:
		timer() {}
		~timer() {}

	public:
		static void init();

		void begin();
		real end();

	private:
		int64_t m_begin;
	};
}