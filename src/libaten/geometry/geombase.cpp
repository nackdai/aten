#include "geometry/geombase.h"

namespace aten
{
    std::atomic<int32_t> geombase::g_id(0);

    geombase::geombase()
    {
        m_geomid = g_id.fetch_add(1);
    }
}
