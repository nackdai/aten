#pragma once

#include <atomic>

#include "types.h"

namespace aten
{
    class NoHitableMesh {
    protected:
        static std::atomic<int32_t> g_id;

        NoHitableMesh();
        virtual ~NoHitableMesh() {}

    public:
        int32_t getGeomId() const
        {
            return m_geomid;
        }

    protected:
        int32_t m_geomid{ -1 };
    };
}
