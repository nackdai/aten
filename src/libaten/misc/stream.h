#pragma once

#include "types.h"

namespace aten {
	class IStream {
	protected:
		IStream() {}
		virtual ~IStream() {}

	public:
		virtual uint32_t write(void* p, uint32_t size) = 0;
		virtual uint32_t read(void* p, uint32_t size) = 0;
	};
}