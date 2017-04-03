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

	static const char* g_types[] = {
		"emissive",
		"lambert",
		"specular",
		"refraction",
		"blinn",
		"beckman",
		"ggx",
		"disney_brdf",
		"oren_nayar",
		"toon",
	};

	bool MaterialLoader::addCreator(std::string type, MaterialCreator creator)
	{
		// Check if type is as same as default type.
		for (auto t : g_types) {
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
	//	- albedomap : albedo texture [string]
	//  - normalmap : normal texture [string]
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
	static aten::PolymorphicValue getValue(const tinyxml2::XMLAttribute* a)
	{
		AT_ASSERT(false);
		PolymorphicValue ret;
		return ret;
	}

	template <>
	static aten::PolymorphicValue getValue<vec3>(const tinyxml2::XMLAttribute* a)
	{
		aten::PolymorphicValue v;

		std::string text(a->Value());

		std::vector<std::string> values;
		int num = split(text, values, ' ');

		for (int i = 0; i < std::min<int>(num, 3); i++) {
			v.val.v[i] = (real)atof(values[i].c_str());
		}

		return std::move(v);
	}

	template <>
	static aten::PolymorphicValue getValue<real>(const tinyxml2::XMLAttribute* a)
	{
		aten::PolymorphicValue v;
		v.val.f = a->DoubleValue();
		return std::move(v);
	}

	template <>
	static aten::PolymorphicValue getValue<texture*>(const tinyxml2::XMLAttribute* a)
	{
		auto s = a->Value();

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

	using GetValueFromFile = std::function<aten::PolymorphicValue(const tinyxml2::XMLAttribute*)>;
#endif

	enum _MtrlParamType {
		Vec3,
		Texture,
		Double,

		Num,
	};

	static GetValueFromFile g_funcGetValueFromFile[] = {
		getValue<vec3>,
		getValue<texture*>,
		getValue<real>,
	};
	C_ASSERT(AT_COUNTOF(g_funcGetValueFromFile) == _MtrlParamType::Num);

	std::map<std::string, _MtrlParamType> g_paramtypes = {
		std::pair<std::string, _MtrlParamType>("color", _MtrlParamType::Vec3),
		std::pair<std::string, _MtrlParamType>("albedomap", _MtrlParamType::Texture),
		std::pair<std::string, _MtrlParamType>("normalmap", _MtrlParamType::Texture),
		std::pair<std::string, _MtrlParamType>("roughnessmap", _MtrlParamType::Texture),
		std::pair<std::string, _MtrlParamType>("ior", _MtrlParamType::Double),
		std::pair<std::string, _MtrlParamType>("shininess", _MtrlParamType::Double),
		std::pair<std::string, _MtrlParamType>("roughness", _MtrlParamType::Double),
		std::pair<std::string, _MtrlParamType>("metallic", _MtrlParamType::Double),
		std::pair<std::string, _MtrlParamType>("subsurface", _MtrlParamType::Double),
		std::pair<std::string, _MtrlParamType>("specular", _MtrlParamType::Double),
		std::pair<std::string, _MtrlParamType>("roughness", _MtrlParamType::Double),
		std::pair<std::string, _MtrlParamType>("specularTint", _MtrlParamType::Double),
		std::pair<std::string, _MtrlParamType>("anisotropic", _MtrlParamType::Double),
		std::pair<std::string, _MtrlParamType>("sheen[float", _MtrlParamType::Double),
		std::pair<std::string, _MtrlParamType>("sheenTint", _MtrlParamType::Double),
		std::pair<std::string, _MtrlParamType>("clearcoat", _MtrlParamType::Double),
		std::pair<std::string, _MtrlParamType>("clearcoatGloss", _MtrlParamType::Double),
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
	//	- albedomap : albedo texture [string]
	//  - normalmap : normal texture [string]
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

	void MaterialLoader::load(const std::string& path)
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
		}

		auto root = xml.FirstChildElement("root");
		if (root) {
			for (auto elem = root->FirstChildElement("material"); elem != nullptr; elem = elem->NextSiblingElement("material")) {
				material* mtrl = nullptr;
				std::string mtrlName;
				std::string mtrlType;
				Values mtrlValues;

				for (auto attr = elem->FirstAttribute(); attr != nullptr; attr = attr->Next()) {
					std::string attrName(attr->Name());

					if (attrName == "name") {
						mtrlName = attr->Value();

						// Check if there is same name material.
						mtrl = AssetManager::getMtrl(mtrlName);

						if (mtrl) {
							AT_PRINTF("There is same tag material. [%s]\n", mtrlName);
							break;
						}
					}
					else if (attrName == "type") {
						mtrlType = attr->Value();
					}
					else {
						// Get parameter type by parameter name.
						auto itParamType = g_paramtypes.find(attrName);

						if (itParamType != g_paramtypes.end()) {
							auto paramType = itParamType->second;
							auto funcGetValue = g_funcGetValueFromFile[paramType];

							// Get value from json.
							auto value = funcGetValue(attr);

							mtrlValues.add(attrName, value);
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
	}
#endif

	MaterialLoader::MaterialCreator g_funcs[] = {
		[](Values& values) { return new emissive(values); },			// emissive
		[](Values& values) { return new lambert(values); },				// lambert
		[](Values& values) { return new specular(values); },			// specular
		[](Values& values) { return new refraction(values); },			// refraction
		[](Values& values) { return new MicrofacetBlinn(values); },		// blinn
		[](Values& values) { return new MicrofacetBeckman(values); },	// beckman
		[](Values& values) { return new MicrofacetGGX(values); },		// ggx
		[](Values& values) { return new DisneyBRDF(values); },			// disney_brdf
		[](Values& values) { return new OrenNayar(values); },			// oren_nayar
		[](Values& values) { return new toon(values); },				// toon
	};

	C_ASSERT(AT_COUNTOF(g_types) == AT_COUNTOF(g_funcs));

	material* MaterialLoader::create(std::string type, Values& values)
	{
		// Check if default creators are registered.
		if (g_creators.find(g_types[0]) == g_creators.end()) {
			// Register default type.
			for (int i = 0; i < AT_COUNTOF(g_types); i++) {
				g_creators.insert(std::pair<std::string, MaterialCreator>(g_types[i], g_funcs[i]));
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
