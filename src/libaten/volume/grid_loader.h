#include <filesystem>
#include <optional>
#include <vector>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridHandle.h>

#include "defs.h"
#include "misc/tuple.h"

namespace aten {
    inline bool CheckGridMetaData(std::string_view nvdb)
    {
        std::filesystem::path p = nvdb;

        if (!std::filesystem::exists(p)) {
            AT_ASSERT(false);
            AT_PRINTF("%s doesn't exist.", nvdb.data());
            return false;
        }

        try {
            auto list = nanovdb::io::readGridMetaData(nvdb.data());
            if (list.size() != 1) {
                // TODO
                // Support only one grid.
                AT_PRINTF("Support only one grid\n");
                return false;
            }

            if (list[0].gridName != "density") {
                AT_PRINTF("Not denstity grid. Allow only density grid\n");
                return false;
            }

            return true;
        }
        catch (const std::exception& e) {
            AT_PRINTF("An exception occurred: %s\n", e.what());
            return false;
        }
    }

    inline std::optional<nanovdb::GridHandle<nanovdb::HostBuffer>> LoadGrid(
        std::string_view nvdb,
        std::vector<nanovdb::FloatGrid*>& grids)
    {
        if (!aten::CheckGridMetaData(nvdb)) {
            return std::nullopt;
        }

        try {
            auto grid_handle = nanovdb::io::readGrid<nanovdb::HostBuffer>(nvdb.data());
            auto* grid = grid_handle.grid<float>();
            grids.emplace_back(grid);
            return grid_handle;
        }
        catch (const std::exception& e) {
            AT_PRINTF("An exception occurred: %s\n", e.what());
            return std::nullopt;
        }
    }
}
