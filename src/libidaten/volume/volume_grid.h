#pragma once

#include <nanovdb/NanoVDB.h>

namespace idaten
{
    class GridHolder {
    public:
        GridHolder() = default;
        ~GridHolder() = default;

        GridHolder(const GridHolder&) = default;
        GridHolder(GridHolder&&) = default;

        GridHolder& operator=(const GridHolder&) = delete;
        GridHolder& operator=(GridHolder&&) = delete;

        AT_DEVICE_API nanovdb::FloatGrid* GetGrid(int32_t idx) const
        {
            if (grids_) {
                if (0 <= idx && idx < num_) {
                    return grids_[idx];
                }
            }
            return nullptr;
        }

        void AssignGrids(nanovdb::FloatGrid** grids, int32_t num)
        {
            grids_ = grids;
            num_ = num;
        }

        bool IsGridsAssigned() const
        {
            return grids_ != nullptr;
        }

        using FloatGridPtr = nanovdb::FloatGrid*;

    protected:
        nanovdb::FloatGrid** grids_{ nullptr };
        int32_t num_{ 0 };
    };

}
