#pragma once

#include "defs.h"
#include "types.h"
#include "material/material.h"

namespace aten
{
    struct MtrlExportInfo {
        std::string name;
        aten::MaterialParameter param;

        MtrlExportInfo() = default;

        MtrlExportInfo(const char* n, const aten::MaterialParameter& p)
            : name(n), param(p)
        {}
    };

    class MaterialExporter {
    private:
        MaterialExporter() = delete;
        ~MaterialExporter() = delete;

    public:
        static bool exportMaterial(
            std::string_view lpszOutFile,
            const std::vector<MtrlExportInfo>& mtrls);
    };
}
