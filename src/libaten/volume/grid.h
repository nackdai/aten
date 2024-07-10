#pragma once

#include <nanovdb/NanoVDB.h>

#include "defs.h"

namespace aten {
    class Grid : public nanovdb::FloatGrid {};

    class context;

    void ConvertGridToMeshes(
        aten::context& ctxt,
        const Grid* grid);
}
