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

namespace AT_NAME {
    class Grid {
    public:
        static AT_DEVICE_API std::optional<aten::tuple<float, float>> ClipRayByGridBoundingBox(
            const aten::ray& ray,
            const nanovdb::FloatGrid* grid);

        static AT_DEVICE_API float GetValueInGrid(const aten::vec3& p, const nanovdb::FloatGrid* grid);

        Grid() = default;
        ~Grid() = default;

        Grid(const Grid&) = default;
        Grid(Grid&&) = default;

        Grid& operator=(const Grid&) = delete;
        Grid& operator=(Grid&&) = delete;

        void Clear()
        {
#ifdef __AT_CUDA__
            // Nothing to do.
#else
            grids_.clear();
#endif
        }

        AT_DEVICE_API nanovdb::FloatGrid* GetGrid(int32_t idx) const
        {
#ifdef __AT_CUDA__
            if (grids_) {
                return grids_[idx];
            }
            return nullptr;
#else
            if (0 <= idx && idx < grids_.size()) {
                return grids_[idx];
            }
            return nullptr;
#endif
        }

        int32_t AddGrid(nanovdb::FloatGrid* grid)
        {
#ifdef __AT_CUDA__
            return 0;
#else
            grids_.emplace_back(grid);
            int32_t ret = static_cast<int32_t>(grids_.size() - 1);
            return ret;
#endif
        }

        void AssignGrids(nanovdb::FloatGrid** grid, size_t num)
        {
#ifdef __AT_CUDA__
            grids_ = grid;
            num_ = num;
#else
            AT_ASSERT(false);
#endif
        }

    private:
#ifdef __AT_CUDA__
        nanovdb::FloatGrid** grids_{ nullptr };
        size_t num_{ 0 };
#else
        std::vector<nanovdb::FloatGrid*> grids_;
#endif
    };
}
