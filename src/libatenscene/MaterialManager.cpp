#include "MaterialManager.h"
#include "ImageLoader.h"
#include "misc/utility.h"
#include "picojson.h"

namespace aten {
	std::map<std::string, MaterialManager::MaterialCreator> g_creators;
	std::map<std::string, material*> g_mtrls;

	static const char* g_types[] = {
		"emissive",
		"lambert",
		"specular",
		"refraction",
		"blinn",
		"beckman",
		"ggx",
		"disney_brdf",
	};

	bool MaterialManager::addCreator(std::string type, MaterialCreator creator)
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

	bool MaterialManager::addMaterial(std::string tag, material* mtrl)
	{
		auto it = g_mtrls.find(tag);

		if (it == g_mtrls.end()) {
			g_mtrls.insert(std::pair<std::string, material*>(tag, mtrl));
			return true;
		}

		return false;
	}

	material* MaterialManager::load(std::string path)
	{
		std::string pathname;
		std::string extname;
		std::string filename;

		getStringsFromPath(
			path,
			pathname,
			extname,
			filename);

		auto mtrl = load(filename, path);

		return mtrl;
	}

	// NOTE
	// Format
	// {
	//		<material_type> [
	//			<param_name> : <value>
	//		]
	// }
	// param_name
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

		int num = std::min<int>(3, a.size());

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
		auto s = val.get< std::string>();

		aten::PolymorphicValue v;

		texture* tex = ImageLoader::load(s);
		v.val.p = tex;

		return std::move(v);
	}

	enum _MtrlParamType {
		Vec3,
		Texture,
		Double,

		Num,
	};

	using GetValueFromJson = std::function<aten::PolymorphicValue(picojson::value&)>;

	static GetValueFromJson g_funcGetValueFromJson[] = {
		getValue<vec3>,
		getValue<texture*>,
		getValue<real>,
	};
	C_ASSERT(AT_COUNTOF(g_funcGetValueFromJson) == _MtrlParamType::Num);

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

	material* MaterialManager::load(std::string tag, std::string path)
	{
		// Check if there is same name material.
		auto mtrl = get(tag);
		if (mtrl) {
			AT_ASSERT(false);
			AT_PRINTF("There is same tag material. [%s]\n", tag);
			return mtrl;
		}

		std::vector<char> filechars;

		// Read json text.
		FILE* fp = fopen(path.c_str(), "rt");
		{
			fseek(fp, 0, SEEK_END);
			auto size = ftell(fp);
			fseek(fp, 0, SEEK_SET);
			filechars.resize(size + 1);
			fread(&filechars[0], sizeof(char), size, fp);
			fclose(fp);
		}

		std::string strJson(&filechars[0]);

		// Parse json.
		picojson::value json;
		auto err = picojson::parse(json, strJson);

		if (err.empty()) {
			auto& obj = json.get<picojson::object>();

			auto it = obj.begin();
			auto mtrlName = it->first;

			auto& params = obj[mtrlName].get<picojson::object>();

			Values mtrlValues;
			
			// Traverse parameters.
			for (auto it = params.begin(); it != params.end(); it++) {
				auto paramName = it->first;
				auto& jsonVal = it->second;

				// Get parameter type by parameter name.
				auto itParamType = g_paramtypes.find(paramName);

				if (itParamType != g_paramtypes.end()) {
					auto paramType = itParamType->second;

					auto funcGetValue = g_funcGetValueFromJson[paramType];

					// Get value from json.
					auto value = funcGetValue(jsonVal);

					mtrlValues.add(paramName, value);
				}
				else {
					AT_ASSERT(false);
					AT_PRINTF("%s is not suppoorted in %s\n", paramName, mtrlName);
				}
			}

			// Create material;
			mtrl = create(mtrlName, mtrlValues);

			if (mtrl) {
				addMaterial(tag, mtrl);
			}
		}
		else {
			AT_ASSERT(false);
			AT_PRINTF("Json parse err [%s]\n", err.c_str());
		}

		AT_ASSERT(mtrl);
		return mtrl;
	}

	material* MaterialManager::get(std::string tag)
	{
		material* mtrl = nullptr;

		auto itMtrl = g_mtrls.find(tag);
		if (itMtrl != g_mtrls.end()) {
			mtrl = itMtrl->second;
		}

		return mtrl;
	}

	MaterialManager::MaterialCreator g_funcs[] = {
		[](Values& values) { return new emissive(values); },			// emissive
		[](Values& values) { return new lambert(values); },				// lambert
		[](Values& values) { return new specular(values); },			// specular
		[](Values& values) { return new refraction(values); },			// refraction
		[](Values& values) { return new MicrofacetBlinn(values); },		// blinn
		[](Values& values) { return new MicrofacetBeckman(values); },	// beckman
		[](Values& values) { return new MicrofacetGGX(values); },		// ggx
		[](Values& values) { return new DisneyBRDF(values); },			// disney_brdf
	};

	C_ASSERT(AT_COUNTOF(g_types) == AT_COUNTOF(g_funcs));

	material* MaterialManager::create(std::string type, Values& values)
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
