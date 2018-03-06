#include "MaterialLoader.h"
#include "ImageLoader.h"
#include "AssetManager.h"
#include "utility.h"

#ifdef USE_JSON
#include "picojson.h"
#else
#include "tinyxml2.h"
#endif

namespace aten {
	std::map<std::string, MaterialLoader::MaterialCreator> g_creators;

	static std::string g_base;

	void MaterialLoader::setBasePath(const std::string& base)
	{
		g_base = removeTailPathSeparator(base);
	}

	bool MaterialLoader::addCreator(std::string type, MaterialCreator creator)
	{
		const auto& typeNames = aten::material::getMaterialTypeName();

		// Check if type is as same as default type.
		for (auto t : typeNames) {
			if (type == t) {
				AT_ASSERT(false);
				AT_PRINTF("Same as default type [%s]\n", t);
				return false;
			}
		}

		auto it = g_creators.find(type);

		if (it == g_creators.end()) {
			g_creators.insert(std::pair<std::string, MaterialCreator>(type, creator));
			return true;
		}

		return false;
	}

#ifdef USE_JSON
	// NOTE
	// Format
	// {
	//		<tag> : {
	//			<param_name> : <value>
	//		}
	// }
	// param_name
	//  - type : material type [string]. If type is not specified, default type is "lambert".
	//  - color : base color, emissive color, albedo color etc... [float3]
	//	- albedoMap : albedo texture [string]
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

	template <typename TYPE>
	static aten::PolymorphicValue getValue(picojson::value& v)
	{
		AT_ASSERT(false);
		PolymorphicValue ret;
		return ret;
	}

	template <>
	static aten::PolymorphicValue getValue<vec3>(picojson::value& val)
	{
		auto a = val.get<picojson::array>();

		int num = std::min<int>(3, (int)a.size());

		aten::PolymorphicValue v;

		for (int i = 0; i < num; i++) {
			v.val.v[i] = a[i].get<double>();
		}

		return std::move(v);
	}

	template <>
	static aten::PolymorphicValue getValue<real>(picojson::value& val)
	{
		aten::PolymorphicValue v;
		v.val.f = val.get<double>();
		return std::move(v);
	}

	template <>
	static aten::PolymorphicValue getValue<texture*>(picojson::value& val)
	{
		auto s = val.get<std::string>();

		std::string pathname;
		std::string extname;
		std::string filename;

		getStringsFromPath(
			s,
			pathname,
			extname,
			filename);

		auto tex = ImageLoader::load(s);

		aten::PolymorphicValue v;
		v.val.p = tex;

		return std::move(v);
	}

	using GetValueFromFile = std::function<aten::PolymorphicValue(picojson::value&)>;
#else
	template <typename TYPE>
	static aten::PolymorphicValue getValue(const tinyxml2::XMLElement* e)
	{
		AT_ASSERT(false);
		PolymorphicValue ret;
		return ret;
	}

	template <>
	static aten::PolymorphicValue getValue<vec3>(const tinyxml2::XMLElement* e)
	{
		aten::PolymorphicValue v;

		std::string text(e->GetText());

		std::vector<std::string> values;
		int num = split(text, values, ' ');

		for (int i = 0; i < std::min<int>(num, 3); i++) {
			v.val.v[i] = (real)atof(values[i].c_str());
		}

		return std::move(v);
	}

	template <>
	static aten::PolymorphicValue getValue<real>(const tinyxml2::XMLElement* e)
	{
		aten::PolymorphicValue v;
		v.val.f = (real)e->DoubleText();
		return std::move(v);
	}

	template <>
	static aten::PolymorphicValue getValue<texture*>(const tinyxml2::XMLElement* e)
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

		auto tex = ImageLoader::load(s);

		aten::PolymorphicValue v;
		v.val.p = tex;

		return std::move(v);
	}

	using GetValueFromFile = std::function<aten::PolymorphicValue(const tinyxml2::XMLElement*)>;
#endif

	static GetValueFromFile g_funcGetValueFromFile[] = {
		getValue<vec3>,
		getValue<texture*>,
		getValue<real>,
	};
	C_ASSERT(AT_COUNTOF(g_funcGetValueFromFile) == (int)MtrlParamType::Num);

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

#ifdef USE_JSON
	void MaterialLoader::load(const std::string& path)
	{
		std::string fullpath = path;
		if (!g_base.empty()) {
			fullpath = g_base + "/" + fullpath;
		}

		std::vector<char> filechars;

		// Read json text.
		FILE* fp = fopen(fullpath.c_str(), "rt");
		{
			fseek(fp, 0, SEEK_END);
			auto size = ftell(fp);
			fseek(fp, 0, SEEK_SET);
			filechars.resize(size + 1);
			fread(&filechars[0], sizeof(char), size, fp);
			fclose(fp);
		}

		std::string strJson(&filechars[0]);

		onLoad(strJson);
	}

	void MaterialLoader::onLoad(const std::string& strJson)
	{
		// Parse json.
		picojson::value json;
		auto err = picojson::parse(json, strJson);

		if (err.empty()) {
			auto& objs = json.get<picojson::object>();

			for (auto it = objs.begin(); it != objs.end(); it++) {
				auto mtrlName = it->first;

				// Check if there is same name material.
				auto mtrl = AssetManager::getMtrl(mtrlName);

				if (mtrl) {
					AT_PRINTF("There is same tag material. [%s]\n", mtrlName.c_str());
				}
				else {
					auto& params = objs[mtrlName].get<picojson::object>();

					// Search material type.
					std::string mtrlType;
					Values mtrlValues;

					// Traverse parameters.
					for (auto it = params.begin(); it != params.end(); it++) {
						auto paramName = it->first;
						auto& jsonVal = it->second;

						if (paramName == "type") {
							mtrlType = jsonVal.get<std::string>();
						}
						else {
							// Get parameter type by parameter name.
							auto itParamType = g_paramtypes.find(paramName);

							if (itParamType != g_paramtypes.end()) {
								auto paramType = itParamType->second;

								auto funcGetValue = g_funcGetValueFromFile[paramType];

								// Get value from json.
								auto value = funcGetValue(jsonVal);

								mtrlValues.add(paramName, value);
							}
							else {
								AT_ASSERT(false);
								AT_PRINTF("[%s] is not suppoorted in [%s]\n", paramName.c_str(), mtrlName.c_str());
							}
						}
					}

					if (mtrlType.empty()) {
						AT_PRINTF("Material type is not specified in [%s]\n", mtrlName.c_str());
						mtrlType = "lambert";
					}

					// Create material;
					mtrl = create(mtrlType, mtrlValues);

					if (mtrl) {
						AssetManager::registerMtrl(mtrlName, mtrl);
					}
					else {
						AT_ASSERT(false);
						AT_PRINTF("Failed to create material : type[%s] name[%s]\n", mtrlType.c_str(), mtrlName.c_str());
					}
				}
			}
		}
		else {
			AT_ASSERT(false);
			AT_PRINTF("Json parse err [%s]\n", err.c_str());
		}
	}
#else
	// NOTE
	// Format
	// <root>
	//	<material
	//		<param_name>=<vales>
	//	/>
	// </root>
	// param_name
	//  - name : material name [string].
	//  - type : material type [string]. If type is not specified, default type is "lambert".
	//  - color : base color, emissive color, albedo color etc... [float3]
	//	- albedoMap : albedo texture [string]
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

	bool MaterialLoader::load(const std::string& path)
	{
		std::string fullpath = path;
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
			onLoad(root);
		}
		else {
			// TODO
			// throw exception.
			AT_ASSERT(false);
			return false;
		}

		return true;
	}

	void MaterialLoader::onLoad(const void* xmlRoot)
	{
		const tinyxml2::XMLElement* root = (const tinyxml2::XMLElement*)xmlRoot;

		for (auto elem = root->FirstChildElement("material"); elem != nullptr; elem = elem->NextSiblingElement("material")) {
			material* mtrl = nullptr;
			std::string mtrlName;
			std::string mtrlType;
			Values mtrlValues;

			for (auto child = elem->FirstChildElement(); child != nullptr; child = child->NextSiblingElement()) {
				std::string paramName(child->Name());

				if (paramName == "name") {
					mtrlName = child->GetText();

					// Check if there is same name material.
					mtrl = AssetManager::getMtrl(mtrlName);

					if (mtrl) {
						AT_PRINTF("There is same tag material. [%s]\n", mtrlName);
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
						auto funcGetValue = g_funcGetValueFromFile[(int)paramType];

						// Get value from json.
						auto value = funcGetValue(child);

						mtrlValues.add(paramName, value);
					}
				}
			}

			if (!mtrl) {
				if (mtrlType.empty()) {
					AT_PRINTF("Material type is not specified in [%s]\n", mtrlName.c_str());
					mtrlType = "lambert";
				}

				// Create material;
				mtrl = create(mtrlType, mtrlValues);

				if (mtrl) {
					AssetManager::registerMtrl(mtrlName, mtrl);
				}
				else {
					AT_ASSERT(false);
					AT_PRINTF("Failed to create material : type[%s] name[%s]\n", mtrlType.c_str(), mtrlName.c_str());
				}
			}
		}
	}
#endif

	MaterialLoader::MaterialCreator g_funcs[] = {
		[](Values& values) { return new emissive(values); },			// emissive
		[](Values& values) { return new lambert(values); },				// lambert
		[](Values& values) { return new OrenNayar(values); },			// oren_nayar
		[](Values& values) { return new specular(values); },			// specular
		[](Values& values) { return new refraction(values); },			// refraction
		[](Values& values) { return new MicrofacetBlinn(values); },		// blinn
		[](Values& values) { return new MicrofacetBeckman(values); },	// beckman
		[](Values& values) { return new MicrofacetGGX(values); },		// ggx
		[](Values& values) { return new DisneyBRDF(values); },			// disney_brdf
		[](Values& values) { return new toon(values); },				// toon
		[](Values& values) { return nullptr; },							// layer
	};
	C_ASSERT(AT_COUNTOF(g_funcs) == (int)aten::MaterialType::MaterialTypeMax);

	material* MaterialLoader::create(std::string type, Values& values)
	{
		const auto& defaultMtrlNames = aten::material::getMaterialTypeName();

		// Check if default creators are registered.
		if (g_creators.empty()
			|| g_creators.find(defaultMtrlNames[0]) == g_creators.end())
		{
			// Register default type.
			for (int i = 0; i < defaultMtrlNames.size(); i++) {
				g_creators.insert(std::pair<std::string, MaterialCreator>(defaultMtrlNames[i], g_funcs[i]));
			}
		}

		material* mtrl = nullptr;

		auto it = g_creators.find(type);

		if (it != g_creators.end()) {
			auto creator = it->second;
			mtrl = creator(values);
		}

		return mtrl;
	}
}
