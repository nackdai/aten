#include "MtrlExporter.h"
#include "FileOutputStream.h"

#include "tinyxml2.h"

bool MtrlExporter::exportMaterial(
    const char* lpszOutFile,
    aten::FbxImporter* pImporter)
{
    bool ret = true;

	tinyxml2::XMLDocument xmlDoc;
	
	auto xmlRoot = xmlDoc.NewElement("root");

	auto mtrlNum = pImporter->getMaterialNum();

	char buf[128] = { 0 };

	for (uint32_t i = 0; i < mtrlNum; i++) {
		MaterialInfo mtrl;

		if (pImporter->getMaterial(i, mtrl)) {
			auto xmlMtrlElement = xmlDoc.NewElement("material");
			xmlMtrlElement->SetAttribute("name", mtrl.name.c_str());
			xmlMtrlElement->SetAttribute("type", "lambert");	// Fixed type.
			
			if (!mtrl.tex.empty()) {
				// TODO
				// レイヤーテクスチャは許していない.
				bool isExportedAlbedo = false;
				bool isExportedNormal = false;

				for (const auto& tex : mtrl.tex) {
					if (tex.type.isNormal && !isExportedNormal) {
						xmlMtrlElement->SetAttribute("normal", tex.name.c_str());
						isExportedNormal = true;
					}
					else if (tex.type.isSpecular) {
						// TODO
					}
					else if (!isExportedAlbedo) {
						xmlMtrlElement->SetAttribute("albedo", tex.name.c_str());
						isExportedAlbedo = true;
					}
				}
			}

			// TODO
			for (const auto& param : mtrl.params) {
				if (param.name == "diffuse") {
					sprintf(buf, "%.3f %.3f %.3f\0", param.values[0], param.values[1], param.values[2]);
					xmlMtrlElement->SetAttribute("color", buf);
				}
			}

			xmlRoot->InsertEndChild(xmlMtrlElement);
		}
	}

	xmlDoc.InsertFirstChild(xmlRoot);

	auto xmlRes = xmlDoc.SaveFile(lpszOutFile);
	ret = (xmlRes == tinyxml2::XML_SUCCESS);

    return true;
}
