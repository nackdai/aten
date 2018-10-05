#pragma once

#include "types.h"

#include <atomic>

namespace aten
{
    class geombase {
    protected:
        static std::atomic<int> g_id;

        geombase();
        virtual ~geombase() {}

    public:
        int getGeomId() const
        {
            return m_geomid;
        }

    protected:
        int m_geomid{ -1 };
    };

    template <typename INHERIT>
    class geom : public INHERIT, public geombase {
    protected:
        geom() {}
        virtual ~geom() {}

    public:
        virtual int geomid() const override
        {
            return m_geomid;
        }
    };
}
