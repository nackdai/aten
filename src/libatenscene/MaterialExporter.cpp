#include "tinyxml2.h"

#include "MaterialExporter.h"
#include "aten.h"

namespace aten
{
    class MaterialParamExporter : public aten::IMaterialParamEditor {
    public:
        MaterialParamExporter(
            tinyxml2::XMLDocument& xmlDoc,
            tinyxml2::XMLElement* xmlElem)
            : m_xmlDoc(xmlDoc), m_xmlElem(xmlElem)
        {}
        virtual ~MaterialParamExporter() {}

    public:
        virtual bool edit(const char* name, real& param, real _min = real(0), real _max = real(1)) override final
        {
            char buf[32];
            snprintf(buf, AT_COUNTOF(buf), "%.3f", param);

            auto xmlMtrlAttribElem = m_xmlDoc.NewElement(name);
            xmlMtrlAttribElem->SetText(buf);
            m_xmlElem->InsertEndChild(xmlMtrlAttribElem);

            return true;
        }

        virtual bool edit(const char* name, aten::vec3& param) override final
        {
            char buf[64];
            snprintf(buf, AT_COUNTOF(buf), "%.3f %.3f %.3f", param.x, param.y, param.z);

            auto xmlMtrlAttribElem = m_xmlDoc.NewElement(name);
            xmlMtrlAttribElem->SetText(buf);
            m_xmlElem->InsertEndChild(xmlMtrlAttribElem);

            return true;
        }

        virtual bool edit(const char* name, aten::vec4& param) override final
        {
            char buf[64];
            snprintf(
                buf, AT_COUNTOF(buf),
                "%.3f %.3f %.3f %.3f",
                param.x, param.y, param.z, param.w);

            auto xmlMtrlAttribElem = m_xmlDoc.NewElement(name);
            xmlMtrlAttribElem->SetText(buf);
            m_xmlElem->InsertEndChild(xmlMtrlAttribElem);

            return true;
        }

        virtual void edit(const char* name, const char* str) override final
        {
            std::string s(str);

            if (!s.empty()) {
                auto xmlMtrlAttribElem = m_xmlDoc.NewElement(name);
                xmlMtrlAttribElem->SetText(str);
                m_xmlElem->InsertEndChild(xmlMtrlAttribElem);
            }
        }

    private:
        tinyxml2::XMLDocument& m_xmlDoc;
        tinyxml2::XMLElement* m_xmlElem{ nullptr };
    };

    bool MaterialExporter::exportMaterial(
        const char* lpszOutFile,
        const std::vector<MtrlExportInfo>& mtrls)
    {
        bool ret = true;

        tinyxml2::XMLDocument xmlDoc;

        auto xmlDecl = xmlDoc.NewDeclaration();
        xmlDoc.InsertFirstChild(xmlDecl);

        auto xmlRoot = xmlDoc.NewElement("root");

        auto mtrlNum = mtrls.size();

        for (uint32_t i = 0; i < mtrlNum; i++) {
            const auto& info = mtrls[i];

            auto xmlMtrlElement = xmlDoc.NewElement("material");

            MaterialParamExporter paramExporter(xmlDoc, xmlMtrlElement);

            {
                auto xmlMtrlAttribElem = xmlDoc.NewElement("name");
                xmlMtrlAttribElem->SetText(info.name.c_str());
                xmlMtrlElement->InsertEndChild(xmlMtrlAttribElem);
            }

            {
                auto xmlMtrlAttribElem = xmlDoc.NewElement("type");
                xmlMtrlAttribElem->SetText(aten::material::getMaterialTypeName(info.param.type));
                xmlMtrlElement->InsertEndChild(xmlMtrlAttribElem);
            }

            auto mtrl = aten::MaterialFactory::createMaterialWithMaterialParameter(
                info.param,
                nullptr, nullptr, nullptr);

            mtrl->edit(&paramExporter);

            // Not use anymore...
            delete mtrl;

            // TODO
            // texture...

            xmlRoot->InsertEndChild(xmlMtrlElement);
        }

        xmlDoc.InsertAfterChild(xmlDecl, xmlRoot);

        auto xmlRes = xmlDoc.SaveFile(lpszOutFile);
        ret = (xmlRes == tinyxml2::XML_SUCCESS);

        return true;
    }

    bool MaterialExporter::exportMaterial(
        const char* lpszOutFile,
        const std::vector<aten::material*>& mtrls)
    {
        std::vector<MtrlExportInfo> mtrlInfos;

        for (const auto mtrl : mtrls) {
            mtrlInfos.push_back(MtrlExportInfo(mtrl->name(), mtrl->param()));
        }

        return exportMaterial(lpszOutFile, mtrlInfos);
    }
}
