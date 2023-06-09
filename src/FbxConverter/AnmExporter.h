#pragma once

#include "defs.h"
#include "types.h"
#include "FbxImporter.h"

class AnmExporter {
private:
    AnmExporter() = delete;
    ~AnmExporter() = delete;

public:
    static bool exportAnm(
        std::string_view lpszOutFile,
        uint32_t nSetIdx,
        aten::FbxImporter* pImporter);
};
