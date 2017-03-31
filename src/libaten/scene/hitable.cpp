#include <atomic>
#include "scene/hitable.h"

namespace aten
{
	// NOTE
	// 0 は予約済みなので、1 から始める.
	static std::atomic<uint32_t> g_id = 0;

	hitable::hitable(const char* name/*= nullptr*/)
		: m_name(name)
	{
		m_id = g_id.fetch_add(1);
	}
}
