#include "MtrlExporter.h"
#include "FileOutputStream.h"

#include "tinyxml2.h"

// TODO
// libatenscene::MaterialExporterに乗り換えたい...

bool MtrlExporter::exportMaterial(
    std::string_view lpszOutFile,
    aten::FbxImporter* pImporter)
{
    bool ret = true;

    tinyxml2::XMLDocument xmlDoc;

    auto xmlDecl = xmlDoc.NewDeclaration();
    xmlDoc.InsertFirstChild(xmlDecl);

    auto xmlRoot = xmlDoc.NewElement("root");

    auto mtrlNum = pImporter->getMaterialNum();

    char buf[128] = { 0 };

    for (uint32_t i = 0; i < mtrlNum; i++) {
        MaterialInfo mtrl;

        if (pImporter->GetMaterial(i, mtrl)) {
            auto xmlMtrlElement = xmlDoc.NewElement("material");

            {
                auto xmlMtrlAttribElem = xmlDoc.NewElement("name");
                xmlMtrlAttribElem->SetText(mtrl.name.c_str());
                xmlMtrlElement->InsertEndChild(xmlMtrlAttribElem);
            }

            {
                // Fixed type.
                auto xmlMtrlAttribElem = xmlDoc.NewElement("type");
                xmlMtrlAttribElem->SetText("lambert");
                xmlMtrlElement->InsertEndChild(xmlMtrlAttribElem);
            }

            if (!mtrl.tex.empty()) {
                // TODO
                // レイヤーテクスチャは許していない.
                bool isExportedAlbedo = false;
                bool isExportedNormal = false;

                for (const auto& tex : mtrl.tex) {
                    if (tex.type.isNormal && !isExportedNormal) {
                        auto xmlMtrlAttribElem = xmlDoc.NewElement("normalMap");
                        xmlMtrlAttribElem->SetText(tex.name.c_str());
                        xmlMtrlElement->InsertEndChild(xmlMtrlAttribElem);
                        isExportedNormal = true;
                    }
                    else if (tex.type.isSpecular) {
                        // TODO
                    }
                    else if (!isExportedAlbedo) {
                        auto xmlMtrlAttribElem = xmlDoc.NewElement("albedoMap");
                        xmlMtrlAttribElem->SetText(tex.name.c_str());
                        xmlMtrlElement->InsertEndChild(xmlMtrlAttribElem);
                        isExportedAlbedo = true;
                    }
                }
            }

            bool isExportedAlbedoColor = false;

            // TODO
            for (const auto& param : mtrl.params) {
                if (param.name == "diffuse") {
                    sprintf(buf, "%.3f %.3f %.3f\0", param.values[0], param.values[1], param.values[2]);

                    auto xmlMtrlAttribElem = xmlDoc.NewElement("baseColor");
                    xmlMtrlAttribElem->SetText(buf);
                    xmlMtrlElement->InsertEndChild(xmlMtrlAttribElem);

                    isExportedAlbedoColor = true;

                    break;
                }
            }

            if (!isExportedAlbedoColor) {
                sprintf(buf, "1.0 1.0 1.0");

                auto xmlMtrlAttribElem = xmlDoc.NewElement("baseColor");
                xmlMtrlAttribElem->SetText(buf);
                xmlMtrlElement->InsertEndChild(xmlMtrlAttribElem);
            }

            xmlRoot->InsertEndChild(xmlMtrlElement);
        }
    }

    xmlDoc.InsertAfterChild(xmlDecl, xmlRoot);

    auto xmlRes = xmlDoc.SaveFile(lpszOutFile.data());
    ret = (xmlRes == tinyxml2::XML_SUCCESS);

    return true;
}
