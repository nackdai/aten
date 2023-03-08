#include "geometry/NoHitableMesh.h"

namespace aten
{
    std::atomic<int32_t> NoHitableMesh::g_mesh_id(0);

    NoHitableMesh::NoHitableMesh()
    {
        mesh_id_ = g_mesh_id.fetch_add(1);
    }
}
