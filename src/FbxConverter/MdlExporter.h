#pragma once

#include "types.h"
#include "FbxImporter.h"

class MdlExporter {
private:
    MdlExporter();
    ~MdlExporter();

public:
    static bool exportMdl(
        uint32_t maxJointMtxNum,
        const char* lpszOutFile,
        aten::FbxImporter* pImporter,
        bool isExportForGPUSkinning = false);
};
