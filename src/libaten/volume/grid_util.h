#pragma once

#include <optional>

#include <nanovdb/NanoVDB.h>

#include "defs.h"
#include "math/ray.h"
#include "misc/tuple.h"

namespace AT_NAME {
    class GridUtil {
    private:
        GridUtil() = default;
        ~GridUtil() = default;

        GridUtil(const GridUtil&) = default;
        GridUtil(GridUtil&&) = default;

        GridUtil& operator=(const GridUtil&) = delete;
        GridUtil& operator=(GridUtil&&) = delete;

    public:
        static AT_DEVICE_API std::optional<aten::tuple<float, float>> ClipRayByGridBoundingBox(
            const aten::ray& ray,
            const nanovdb::FloatGrid* grid);

        static AT_DEVICE_API float GetValueInGrid(const aten::vec3& p, const nanovdb::FloatGrid* grid);
    };
}
