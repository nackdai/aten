#pragma once

#include <vector>

#include "defs.h"

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
