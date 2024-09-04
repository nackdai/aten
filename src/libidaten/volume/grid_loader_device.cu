#include "volume/grid_loader_device.h"
#include "volume/grid_loader.h"

// NOTE:
// In linux, NanoVDB for cuda stuff can't be linked as the undefined reference
// when we include the related header files in main.cpp of the executable.
// We're not sure why we can't do it in linux. clang? CMake?
// On the other hand, if we make library to call NanoVDB for cuda stuff explicitly and link it from the executable,
// it can be built as expected. And, the file doesn't have to be .cpp file but .cu file.
// From these above situation, we implement the API to call NanoVDB cuda related APIs as the library.

namespace idaten {
    std::optional<nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>> LoadGrid(
        std::string_view nvdb,
        std::vector<nanovdb::FloatGrid*>& grids,
        cudaStream_t stream)
    {
        if (!aten::CheckGridMetaData(nvdb)) {
            return std::nullopt;
        }

        try {
            auto grid_handle = nanovdb::io::readGrid<nanovdb::CudaDeviceBuffer>(nvdb.data());
            grid_handle.deviceUpload(stream, true);
            auto* grid = grid_handle.deviceGrid<float>();
            grids.emplace_back(grid);
            return grid_handle;
        }
        catch (const std::exception& e) {
            AT_PRINTF("An exception occurred: %s\n", e.what());
            return std::nullopt;
        }
    }
}
