#include "tinyxml2.h"

#include "MaterialLoader.h"
#include "ImageLoader.h"
#include "utility.h"

namespace aten {
    static std::string g_base;

    void MaterialLoader::setBasePath(const std::string& base)
    {
        g_base = removeTailPathSeparator(base);
    }

    template <class TYPE>
    aten::PolymorphicValue getValue(const tinyxml2::XMLElement* e, aten::context& ctxt)
    {
        AT_ASSERT(false);
        PolymorphicValue ret;
        return ret;
    }

    template <>
    aten::PolymorphicValue getValue<vec3>(const tinyxml2::XMLElement* e, aten::context& ctxt)
    {
        aten::PolymorphicValue val;

        std::string text(e->GetText());

        std::vector<std::string> values;
        int32_t num = split(text, values, ' ');

        aten::vec4 v;
        for (int32_t i = 0; i < std::min<int32_t>(num, 3); i++) {
            v[i] = (float)atof(values[i].c_str());
        }

        val = v;

        return val;
    }

    template <>
    aten::PolymorphicValue getValue<float>(const tinyxml2::XMLElement* e, aten::context& ctxt)
    {
        aten::PolymorphicValue v;
        v = (float)e->DoubleText();
        return v;
    }

    template <>
    aten::PolymorphicValue getValue<texture*>(const tinyxml2::XMLElement* e, aten::context& ctxt)
    {
        auto s = e->GetText();

        std::string pathname;
        std::string extname;
        std::string filename;

        getStringsFromPath(
            s,
            pathname,
            extname,
            filename);

        auto tex = ImageLoader::load(s, ctxt);

        aten::PolymorphicValue v;
        v = tex;

        return v;
    }

    using GetValueFromFile = std::function<aten::PolymorphicValue(const tinyxml2::XMLElement*, aten::context&)>;

    static std::array<GetValueFromFile, static_cast<size_t>(MtrlParamType::Num)> g_funcGetValueFromFile = {
        getValue<vec3>,
        getValue<texture*>,
        getValue<float>,
    };

    std::map<std::string, MtrlParamType> g_paramtypes = {
        std::pair<std::string, MtrlParamType>("baseColor", MtrlParamType::Vec3),
        std::pair<std::string, MtrlParamType>("ior", MtrlParamType::Double),
        std::pair<std::string, MtrlParamType>("roughness", MtrlParamType::Double),
        std::pair<std::string, MtrlParamType>("shininess", MtrlParamType::Double),
        std::pair<std::string, MtrlParamType>("subsurface", MtrlParamType::Double),
        std::pair<std::string, MtrlParamType>("metallic", MtrlParamType::Double),
        std::pair<std::string, MtrlParamType>("specular", MtrlParamType::Double),
        std::pair<std::string, MtrlParamType>("specularTint", MtrlParamType::Double),
        std::pair<std::string, MtrlParamType>("anisotropic", MtrlParamType::Double),
        std::pair<std::string, MtrlParamType>("sheen", MtrlParamType::Double),
        std::pair<std::string, MtrlParamType>("sheenTint", MtrlParamType::Double),
        std::pair<std::string, MtrlParamType>("clearcoat", MtrlParamType::Double),
        std::pair<std::string, MtrlParamType>("clearcoatGloss", MtrlParamType::Double),
        std::pair<std::string, MtrlParamType>("albedoMap", MtrlParamType::Texture),
        std::pair<std::string, MtrlParamType>("normalMap", MtrlParamType::Texture),
        std::pair<std::string, MtrlParamType>("roughnessMap", MtrlParamType::Texture),
    };

    // NOTE
    // Format
    // <root>
    //    <material
    //        <param_name>=<vales>
    //    />
    // </root>
    // param_name
    //  - name : material name [string].
    //  - type : material type [string]. If type is not specified, default type is "Diffuse".
    //  - color : base color, emissive color, albedo color etc... [float3]
    //    - albedoMap : albedo texture [string]
    //  - normalMap : normal texture [string]
    //  - roughnessmap : roughness texture [string]
    //  - ior : index of reflaction [float]
    //  - shininess [float]
    //  - roughness [float]
    //  - metallic [float]
    //  - subsurface [float]
    //  - specular [float]
    //  - roughness [float]
    //  - specularTint [float]
    //  - anisotropic [float]
    //  - sheen [float]
    //  - sheenTint [float]
    //  - clearcoat [float]
    //  - clearcoatGloss [float]

    bool MaterialLoader::load(
        std::string_view path,
        context& ctxt)
    {
        std::string fullpath(path);
        if (!g_base.empty()) {
            fullpath = g_base + "/" + fullpath;
        }

        tinyxml2::XMLDocument xml;
        auto err = xml.LoadFile(fullpath.c_str());
        if (err != tinyxml2::XML_SUCCESS) {
            // TODO
            // throw exception.
            AT_ASSERT(false);
            return false;
        }

        auto root = xml.FirstChildElement("root");
        if (root) {
            onLoad(root, ctxt);
        }
        else {
            // TODO
            // throw exception.
            AT_ASSERT(false);
            return false;
        }

        return true;
    }

    void MaterialLoader::onLoad(
        const void* xmlRoot,
        context& ctxt)
    {
        const tinyxml2::XMLElement* root = (const tinyxml2::XMLElement*)xmlRoot;

        for (auto elem = root->FirstChildElement("material"); elem != nullptr; elem = elem->NextSiblingElement("material")) {
            std::shared_ptr<material> mtrl;
            std::string mtrlName;
            std::string mtrlType;
            Values mtrlValues;

            for (auto child = elem->FirstChildElement(); child != nullptr; child = child->NextSiblingElement()) {
                std::string paramName(child->Name());

                if (paramName == "name") {
                    mtrlName = child->GetText();

                    // Check if there is same name material.
                    mtrl = ctxt.FindMaterialByName(mtrlName);

                    if (mtrl) {
                        AT_PRINTF("There is same tag material. [%s]\n", mtrlName.c_str());
                        break;
                    }
                }
                else if (paramName == "type") {
                    mtrlType = child->GetText();
                }
                else {
                    // Get parameter type by parameter name.
                    auto itParamType = g_paramtypes.find(paramName);

                    if (itParamType != g_paramtypes.end()) {
                        auto paramType = itParamType->second;
                        auto funcGetValue = g_funcGetValueFromFile[(int32_t)paramType];

                        // Get value from json.
                        auto value = funcGetValue(child, ctxt);

                        mtrlValues.add(paramName, value);
                    }
                }
            }

            if (!mtrl) {
                if (mtrlType.empty()) {
                    AT_PRINTF("Material type is not specified in [%s]\n", mtrlName.c_str());
                    mtrlType = "Diffuse";
                }

                // Create material;
                mtrl = create(mtrlName, mtrlType, ctxt, mtrlValues);

                if (!mtrl) {
                    AT_ASSERT(false);
                    AT_PRINTF("Failed to create material : type[%s] name[%s]\n", mtrlType.c_str(), mtrlName.c_str());
                }
            }
        }
    }

    std::shared_ptr<material> MaterialLoader::create(
        std::string_view name,
        const std::string& type,
        context& ctxt,
        Values& values)
    {
        std::shared_ptr<material> mtrl;

        if (aten::material::isDefaultMaterialName(type)) {

            auto mtrlType = aten::material::getMaterialTypeFromMaterialTypeName(type);

            if (aten::material::isValidMaterialType(mtrlType)) {
                mtrl = ctxt.CreateMaterial(name, mtrlType, values);
            }
        }

        return mtrl;
    }
}
