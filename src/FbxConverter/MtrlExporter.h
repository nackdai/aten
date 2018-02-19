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
    bool exportMaterial(
        const char* lpszOutFile,
        aten::FbxImporter* pImporter,
        uint32_t nMtrlIdx);

private:
    bool exportMaterial(
		FileOutputStream& out,
        uint32_t nMtrlIdx,
		aten::FbxImporter* pImporter);
};
