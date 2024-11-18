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
        virtual bool edit(std::string_view name, float& param, float _min, float _max) override final
        {
            char buf[32];
            snprintf(buf, AT_COUNTOF(buf), "%.3f", param);

            auto xmlMtrlAttribElem = m_xmlDoc.NewElement(name.data());
            xmlMtrlAttribElem->SetText(buf);
            m_xmlElem->InsertEndChild(xmlMtrlAttribElem);

            return true;
        }

        bool edit(std::string_view name, bool& param) override final
        {
            AT_ASSERT(false);
            return false;
        }

        virtual bool edit(std::string_view name, aten::vec3& param) override final
        {
            char buf[64];
            snprintf(buf, AT_COUNTOF(buf), "%.3f %.3f %.3f", param.x, param.y, param.z);

            auto xmlMtrlAttribElem = m_xmlDoc.NewElement(name.data());
            xmlMtrlAttribElem->SetText(buf);
            m_xmlElem->InsertEndChild(xmlMtrlAttribElem);

            return true;
        }

        virtual bool edit(std::string_view name, aten::vec4& param) override final
        {
            char buf[64];
            snprintf(
                buf, AT_COUNTOF(buf),
                "%.3f %.3f %.3f %.3f",
                param.x, param.y, param.z, param.w);

            auto xmlMtrlAttribElem = m_xmlDoc.NewElement(name.data());
            xmlMtrlAttribElem->SetText(buf);
            m_xmlElem->InsertEndChild(xmlMtrlAttribElem);

            return true;
        }

        virtual void edit(std::string_view name, std::string_view str) override final
        {
            std::string s(str);

            if (!s.empty()) {
                auto xmlMtrlAttribElem = m_xmlDoc.NewElement(name.data());
                xmlMtrlAttribElem->SetText(str.data());
                m_xmlElem->InsertEndChild(xmlMtrlAttribElem);
            }
        }

        bool edit(std::string_view name, const std::vector<const char*>& elements, int32_t& param) override final
        {
            // TODO
            AT_ASSERT(false);
            return false;
        }

    private:
        tinyxml2::XMLDocument& m_xmlDoc;
        tinyxml2::XMLElement* m_xmlElem{ nullptr };
    };

    bool MaterialExporter::exportMaterial(
        std::string_view lpszOutFile,
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

            auto mtrl = aten::material::CreateMaterialWithMaterialParameter(
                info.param,
                nullptr, nullptr, nullptr);

            mtrl->edit(&paramExporter);

            // TODO
            // texture...

            xmlRoot->InsertEndChild(xmlMtrlElement);
        }

        xmlDoc.InsertAfterChild(xmlDecl, xmlRoot);

        auto xmlRes = xmlDoc.SaveFile(lpszOutFile.data());
        ret = (xmlRes == tinyxml2::XML_SUCCESS);

        return true;
    }
}
