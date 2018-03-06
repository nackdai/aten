#pragma once

#include "defs.h"
#include "types.h"
#include "material/material.h"

namespace aten
{
	struct MtrlExportInfo {
		std::string name;
		aten::MaterialParameter param;

		MtrlExportInfo() {}

		MtrlExportInfo(const char* n, const aten::MaterialParameter& p)
			: name(n), param(p)
		{}
	};

	class MaterialExporter {
	private:
		MaterialExporter();
		~MaterialExporter();

	public:
		static bool exportMaterial(
			const char* lpszOutFile,
			const std::vector<MtrlExportInfo>& mtrls);

		static bool exportMaterial(
			const char* lpszOutFile,
			const std::vector<aten::material*>& mtrls);
	};
}