#include "shape/mesh.h"

namespace aten
{
	std::atomic<int> meshbase::g_id = 0;

	meshbase::meshbase()
	{
		m_meshid = g_id.fetch_add(1);
	}
}
