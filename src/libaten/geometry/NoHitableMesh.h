#pragma once

#include <atomic>

#include "types.h"

namespace aten
{
    class NoHitableMesh {
    protected:
        static std::atomic<int32_t> g_mesh_id;

        NoHitableMesh();
        virtual ~NoHitableMesh() {}

    public:
        int32_t get_mesh_id() const
        {
            return mesh_id_;
        }

    protected:
        int32_t mesh_id_{ -1 };
    };
}
