#pragma once

#include "types.h"
#include "FbxImporter.h"

class MdlExporter {
private:
    MdlExporter() = delete;
    ~MdlExporter() = delete;

public:
    static bool exportMdl(
        uint32_t maxJointMtxNum,
        std::string_view lpszOutFile,
        aten::FbxImporter* pImporter,
        bool isExportForGPUSkinning = false);
};
