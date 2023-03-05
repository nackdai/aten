#pragma once

#include <atomic>

#include "types.h"

namespace aten
{
    class geombase {
    protected:
        static std::atomic<int32_t> g_id;

        geombase();
        virtual ~geombase() {}

    public:
        int32_t getGeomId() const
        {
            return m_geomid;
        }

    protected:
        int32_t m_geomid{ -1 };
    };
}
