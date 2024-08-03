#pragma once

#include <optional>

#include "defs.h"
#include "misc/tuple.h"
#include "geometry/PolygonObject.h"

#ifndef NANOVDB_NANOVDB_H_HAS_BEEN_INCLUDED
namespace nanovdb {
    class FloatGrid;
}
#endif

namespace aten {
    class context;

    /**
     * @brief Getnerate triangle from boudning box of VDB grid.
     *
     * @param[in, out] ctxt Scene context.
     * @param[in] target_mtrl_id Index of associated material to grid in the scene context.
     * @param[in] grid VDB grid.
     * @return Object to store generated triangles.
     */
    std::shared_ptr<aten::PolygonObject> GenerateTrianglesFromGridBoundingBox(
        aten::context& ctxt,
        const int32_t target_mtrl_id,
        const nanovdb::FloatGrid* grid);
}

namespace AT_NAME {
    class Grid {
    public:
        static AT_DEVICE_API std::optional<aten::tuple<float, float>> ClipRayByGridBoundingBox(
            const aten::ray& ray,
            const nanovdb::FloatGrid* grid);

        static AT_DEVICE_API float GetValueInGrid(const aten::vec3& p, const nanovdb::FloatGrid* grid);
    };
}
