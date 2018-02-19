#pragma once

#include "defs.h"
#include "types.h"
#include "FbxImporter.h"

class AnmExporter {
private:
	AnmExporter();
    ~AnmExporter();

public:
    static bool exportAnm(
        const char* lpszOutFile,
        uint32_t nSetIdx,
        aten::FbxImporter* pImporter);
};
