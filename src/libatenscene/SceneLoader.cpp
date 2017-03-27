#include "SceneLoader.h"
#include "utility.h"
#include "picojson.h"

namespace aten
{
	static std::string g_base;

	void SceneLoader::setBasePath(const std::string& base)
	{
		g_base = removeTailPathSeparator(base);
	}

	// NOTE
	// Format
	// {
	//		"textures" : {
	//			<tag> : <path>
	//		}
	//		"materials" : {
	//			<tag> : <path>
	//		}
	//		"objects" : {
	//			<tag> : <path>
	//		}
	//		"scene" : {
	//			"config" : {
	//				"width" : <width_of_resolution>
	//				"height" : <height_of_resolution>
	//				"renderer" {
	//					"type" : <renderer_type>
	//					"spp" : <number_of_samples>
	//					"mutation" : <number_of_mutation>
	//					"mlt" : <number_of_mlt>
	//					"depth" : <max_depth>
	//					"rrdepth" : <depth_of_russian_roulette>
	//				}
	//				"bg" : <envmap_tag> or [<r>, <g>, <b>]
	//				"preproc" : {
	//					<preproc_tag>
	//				}
	//				"postproc" : {
	//					<postproc_tag>
	//				}
	//			]
	//			<object name> : {
	//				"obj" : <object_tag>
	//				"trans" : [<x>, <y>, <z>]
	//				"scale" : <scale>
	//				"rotate" : [<x>, <y>, <z>]
	//				"material" : <material_tag>
	//			}
	//		]
	// }

	using AssetInfoMap = std::map<std::string, std::string>;

	static AssetInfoMap g_matrlpaths;
	static AssetInfoMap g_objpaths;
	static AssetInfoMap g_texturepaths;

	void readAssetInfo(AssetInfoMap& assetPaths, picojson::object& infos)
	{	
		for (auto it = infos.begin(); it != infos.end(); it++) {
			auto tag = it->first;
			auto& jsonVal = it->second;

			if (assetPaths.find(tag) != assetPaths.end()) {
				// Not registered.
				auto path = jsonVal.get< std::string>();

				assetPaths.insert(std::pair<std::string, std::string>(tag, path));
			}
		}
	}

	struct ScenObjInfo {
		std::string objtag;
		vec3 trans;
		vec3 scale;
		vec3 rotate;
		std::string mtrltag;
	};

	std::vector<ScenObjInfo> g_sceneObjInfos;

	void readSceneInfo(picojson::object& infos)
	{
		for (auto it = infos.begin(); it != infos.end(); it++) {
			auto name = it->first;

			// Convert to lower.
			std::transform(name.begin(), name.end(), name.begin(), ::tolower);

			if (name == "config") {
				auto configs = infos["config"].get<picojson::object>();

				for (auto it = configs.begin(); it != configs.end(); it++) {
					auto paramName = it->first;
					auto& jsonVal = it->second;

					// Convert to lower.
					std::transform(paramName.begin(), paramName.end(), paramName.begin(), ::tolower);

					if (paramName == "width") {
					}
					else if (paramName == "height") {
					}
					else if (paramName == "renderer") {
					}
					else if (paramName == "bg") {
					}
					else if (paramName == "preproc") {
					}
					else if (paramName == "postproc") {
					}
				}
			}
			else {

			}
		}
	}

	scene* SceneLoader::load(const std::string& path)
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

		// Parse json.
		picojson::value json;
		auto err = picojson::parse(json, strJson);

		if (err.empty()) {
			auto params = json.get<picojson::object>();

			for (auto it = params.begin(); it != params.end(); it++) {
				auto name = it->first;

				auto values = params[name].get<picojson::object>();

				// Convert to lower.
				std::transform(name.begin(), name.end(), name.begin(), ::tolower);

				if (name == "materials") {
					readAssetInfo(g_matrlpaths, values);
				}
				else if (name == "textures") {
					readAssetInfo(g_texturepaths, values);
				}
				else if (name == "objects") {
					readAssetInfo(g_objpaths, values);
				}
				else if (name == "scene") {

				}
				else {
					// TODO
					// throw exception.
				}
			}
		}

		// TODO
		scene* scene = new aten::AcceledScene<aten::bvh>();

		return scene;
	}
}
