#include "geometry/NoHitableMesh.h"

namespace aten
{
    std::atomic<int32_t> NoHitableMesh::g_id(0);

    NoHitableMesh::NoHitableMesh()
    {
        m_geomid = g_id.fetch_add(1);
    }
}
