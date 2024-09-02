#pragma once

#include <optional>
#include <vector>

#include <nanovdb/NanoVDB.h>

#include "defs.h"
#include "misc/tuple.h"
#include "geometry/PolygonObject.h"

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
    std::shared_ptr<AT_NAME::PolygonObject> GenerateTrianglesFromGridBoundingBox(
        aten::context& ctxt,
        const int32_t target_mtrl_id,
        const nanovdb::FloatGrid* grid);
}

namespace aten {
    class Grid {
    public:
        Grid() = default;
        ~Grid() = default;

        Grid(const Grid&) = default;
        Grid(Grid&&) = default;

        Grid& operator=(const Grid&) = delete;
        Grid& operator=(Grid&&) = delete;

        void Clear()
        {
            grids_.clear();
        }

        nanovdb::FloatGrid* GetGrid(int32_t idx) const
        {
            if (0 <= idx && idx < grids_.size()) {
                return grids_[idx];
            }
            return nullptr;
        }

        int32_t AddGrid(nanovdb::FloatGrid* grid)
        {
            grids_.emplace_back(grid);
            int32_t ret = static_cast<int32_t>(grids_.size() - 1);
            return ret;
        }

        int32_t GetGridsNum() const
        {
            return grids_.size();
        }

        nanovdb::FloatGrid* const* GetGrids() const
        {
            return grids_.data();
        }

    private:
        std::vector<nanovdb::FloatGrid*> grids_;
    };
}
