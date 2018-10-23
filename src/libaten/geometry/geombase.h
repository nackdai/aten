#pragma once

#include <atomic>

#include "types.h"

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
    class geom : public virtual INHERIT, public geombase {
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
