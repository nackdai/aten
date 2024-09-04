#include <optional>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/cuda/CudaDeviceBuffer.h>

namespace idaten {
    std::optional<nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>> LoadGrid(
        std::string_view nvdb,
        std::vector<nanovdb::FloatGrid*>& grids,
        cudaStream_t stream);
}
