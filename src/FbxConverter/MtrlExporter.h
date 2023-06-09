#pragma once

#include "defs.h"
#include "types.h"
#include "FbxImporter.h"

class FileOutputStream;

class MtrlExporter {
private:
    MtrlExporter() = delete;
    ~MtrlExporter() = delete;

public:
    static bool exportMaterial(
        std::string_view lpszOutFile,
        aten::FbxImporter* pImporter);
};
