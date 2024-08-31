#include "kernel/device_scene_context.cuh"

#include "volume/grid.h"

namespace idaten {
    __device__ const AT_NAME::Grid* context::GetGrid() const noexcept
    {
        return grid_holder;
    }
}
