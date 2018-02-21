#pragma once

#include "defs.h"
#include "types.h"
#include "FbxImporter.h"

class FileOutputStream;

class MtrlExporter {
private:
    MtrlExporter();
    ~MtrlExporter();

public:
    static bool exportMaterial(
        const char* lpszOutFile,
        aten::FbxImporter* pImporter);
};
