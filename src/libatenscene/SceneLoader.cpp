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
	//		},
	//		"materials" : {
	//			"path" : <path>,
	//			<tag> : {
	//				<param_name> : <value>
	//			}
	//		},
	//		"objects" : {
	//			<tag> : <path>
	//		},
	//		"scene" : {
	//			"config" : {
	//				"width" : <width_of_resolution>,
	//				"height" : <height_of_resolution>,
	//				"renderer" : {
	//					"type" : <renderer_type>
	//					"spp" : <number_of_samples>
	//					"mutation" : <number_of_mutation>
	//					"mlt" : <number_of_mlt>
	//					"depth" : <max_depth>
	//					"rrdepth" : <depth_of_russian_roulette>
	//				},
	//				"bg" : <envmap_tag> or [<r>, <g>, <b>],
	//				"preproc" : {
	//					<preproc_tag>, ...
	//				},
	//				"postproc" : {
	//					<postproc_tag>, ...
	//				},
	//			},
	//			<object name> : {
	//				"obj" : <object_tag>,
	//				"trans" : [<x>, <y>, <z>],
	//				"scale" : <scale>,
	//				"rotate" : [<x>, <y>, <z>],
	//				"material" : <material_tag>
	//			}
	//		]
	// }

	struct AssetInfo {
		std::string tag;
		std::string info;
	};

	using AssetInfoMap = std::vector<AssetInfo>;

	static AssetInfoMap g_mtrlInfos;
	static AssetInfoMap g_objInfos;
	static AssetInfoMap g_textureInfos;

	void readAssetInfo(AssetInfoMap& assetInfos, picojson::object& infos)
	{	
		for (auto it = infos.begin(); it != infos.end(); it++) {
			assetInfos.push_back(AssetInfo());
			AssetInfo& info = assetInfos[assetInfos.size() - 1];

			info.tag = it->first;

			auto& jsonVal = it->second;
			info.info = jsonVal.get<std::string>();			
		}
	}

	struct SceneObjInfo {
		std::string objtag;
		vec3 trans;
		vec3 scale;
		vec3 rotate;
		std::string mtrltag;
	};

	std::vector<SceneObjInfo> g_sceneObjInfos;

	void readRendererInfo(
		SceneLoader::SceneInfo& info,
		picojson::object& objs)
	{
		for (auto it = objs.begin(); it != objs.end(); it++) {
			auto name = it->first;
			auto& jsonVal = it->second;

			// Convert to lower.
			std::transform(name.begin(), name.end(), name.begin(), ::tolower);

			if (name == "type") {
				info.rendererType = jsonVal.get<std::string>();
				std::transform(info.rendererType.begin(), info.rendererType.end(), info.rendererType.begin(), ::tolower);
			}
			else if (name == "spp") {
				info.dst.sample = (int)jsonVal.get<double>();
			}
			else if (name == "mutation") {
				info.dst.mutation = (int)jsonVal.get<double>();
			}
			else if (name == "mlt") {
				info.dst.mltNum = (int)jsonVal.get<double>();
			}
			else if (name == "depth") {
				info.dst.maxDepth = (int)jsonVal.get<double>();
			}
			else if (name == "rrdepth") {
				info.dst.russianRouletteDepth = (int)jsonVal.get<double>();
			}
		}
	}

	void readSceneInfo(
		SceneLoader::SceneInfo& info,
		picojson::object& objs)
	{
		for (auto it = objs.begin(); it != objs.end(); it++) {
			auto name = it->first;

			// Convert to lower.
			std::transform(name.begin(), name.end(), name.begin(), ::tolower);

			if (name == "config") {
				auto configs = objs["config"].get<picojson::object>();

				for (auto it = configs.begin(); it != configs.end(); it++) {
					auto paramName = it->first;
					auto& jsonVal = it->second;

					// Convert to lower.
					std::transform(paramName.begin(), paramName.end(), paramName.begin(), ::tolower);

					if (paramName == "width") {
						info.dst.width = (int)jsonVal.get<double>();
					}
					else if (paramName == "height") {
						info.dst.height = (int)jsonVal.get<double>();
					}
					else if (paramName == "renderer") {
						auto& rendererObjs = jsonVal.get<picojson::object>();
						readRendererInfo(info, rendererObjs);
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

	SceneLoader::SceneInfo SceneLoader::load(const std::string& path)
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

		SceneInfo info;

		if (err.empty()) {
			auto params = json.get<picojson::object>();

			for (auto it = params.begin(); it != params.end(); it++) {
				auto name = it->first;

				auto values = params[name].get<picojson::object>();

				// Convert to lower.
				std::transform(name.begin(), name.end(), name.begin(), ::tolower);

				if (name == "materials") {
					readAssetInfo(g_mtrlInfos, values);
				}
				else if (name == "textures") {
					readAssetInfo(g_textureInfos, values);
				}
				else if (name == "objects") {
					readAssetInfo(g_objInfos, values);
				}
				else if (name == "scene") {
					readSceneInfo(info, values);
				}
				else {
					// TODO
					// throw exception.
				}
			}
		}

		// TODO
		info.scene = new aten::AcceledScene<aten::bvh>();

		return std::move(info);
	}
}
