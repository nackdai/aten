#include "SceneLoader.h"
#include "utility.h"
#include "ImageLoader.h"
#include "MaterialLoader.h"
#include "ObjLoader.h"
#include "AssetManager.h"

#ifdef USE_JSON
#include "picojson.h"
#else
#include "tinyxml2.h"
#endif

namespace aten
{
	static std::string g_base;

	void SceneLoader::setBasePath(const std::string& base)
	{
		g_base = removeTailPathSeparator(base);
	}

#ifdef USE_JSON
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
	//					<preproc_tag> : {
	//						<param_name> : <value>
	//					},
	//				},
	//				"postproc" : {
	//					<preproc_tag> : {
	//						<param_name> : <value>
	//					},
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

	void readProcInfo(
		std::vector<SceneLoader::ProcInfo>& infos,
		picojson::object& objs)
	{
		for (auto it = objs.begin(); it != objs.end(); it++) {
			SceneLoader::ProcInfo procInfo;

			auto name = it->first;
			auto& jsonVal = it->second;

			// Convert to lower.
			std::transform(name.begin(), name.end(), name.begin(), ::tolower);

			procInfo.type = name;

			auto& params = jsonVal.get<picojson::object>();

			for (auto pit = params.begin(); pit != params.end(); pit++) {
				auto paramName = pit->first;
				auto& val = pit->second;

				// Convert to lower.
				std::transform(paramName.begin(), paramName.end(), paramName.begin(), ::tolower);

				aten::PolymorphicValue param;
				param = val.get<double>();

				procInfo.values.add(paramName, param);
			}

			infos.push_back(procInfo);
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
						auto& preprocObjs = jsonVal.get<picojson::object>();
						readProcInfo(info.preprocs, preprocObjs);
					}
					else if (paramName == "postproc") {
						auto& postprocObjs = jsonVal.get<picojson::object>();
						readProcInfo(info.postprocs, postprocObjs);
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
		else {
			// TODO
			// throw exception.
		}

		// TODO
		info.scene = new aten::AcceledScene<aten::bvh>();

		return std::move(info);
	}
#else
	// NOTE
	// <scene width=<uint> height=<uint>>
	//		<camera 
	//			type=<string>
	//			org=<vec3>
	//			at=<vec3> 
	//			fov=<real>
	//			sensorsize=<real>
	//			dist_sensor_lens=<real>
	//			dist_lens_focus=<real> 
	//			lens_r=<real>
	//			W_scale=<real>
	//		/>
	//		<renderer type=<string> spp=<uint> mutaion=<uint> mlt=<uint> depth=<uint> rrdepth=<uint>/>
	//		<materials>
	//			<material path=<string/>
	//			<material [attributes...]/>
	//		</materials>
	//		<textures>
	//			<texture name=<string> path=<string>/>
	//		</textures>
	//		<objects>
	//			<object name=<string> type=<string> path=<string> trans=<vec3> rotate=<vec3> scale=<float> material=<string>/>
	//		</objects>
	//		<lights>
	//			<light type=<string> color=<vec3> [attributes...]/>
	//		</lights>
	//		<preprocs>
	//			<proc type=<string> [attributes...]/>
	//		</preprocs>
	//		<postprocs>
	//			<proc type=<string> [attributes...]/>
	//		</postprocs>
	// </scene>

	void readTextures(const tinyxml2::XMLElement* root)
	{
		auto texRoot = root->FirstChildElement("textures");

		if (!texRoot) {
			return;
		}

		for (auto elem = texRoot->FirstChildElement("texture"); elem != nullptr; elem = texRoot->NextSiblingElement("texture")) {
			std::string path;
			std::string tag;

			for (auto attr = elem->FirstAttribute(); attr != nullptr; attr = attr->Next()) {
				std::string attrName(attr->Name());

				if (attrName == "path") {
					path = attr->Value();
				}
				else if (attrName == "name") {
					tag = attr->Value();
				}
			}

			if (!tag.empty()) {
				ImageLoader::load(tag, path);
			}
			else {
				ImageLoader::load(path);
			}
		}
	}

	void readMaterials(const tinyxml2::XMLElement* root)
	{
		auto mtrlRoot = root->FirstChildElement("materials");
		
		if (!mtrlRoot) {
			return;
		}

		for (auto elem = mtrlRoot->FirstChildElement("material"); elem != nullptr; elem = mtrlRoot->NextSiblingElement("material")) {
			std::string path;
			std::string tag;

			tinyxml2::XMLDocument xml;
			tinyxml2::XMLElement* root = nullptr;
			tinyxml2::XMLElement* mtrlElem = nullptr;

			for (auto attr = elem->FirstAttribute(); attr != nullptr; attr = attr->Next()) {
				std::string attrName(attr->Name());

				if (attrName == "path") {
					path = attr->Value();
				}
				else {
					if (attrName == "name") {
						tag = attr->Value();
					}

					if (!root) {
						root = xml.NewElement("root");
						xml.InsertEndChild(root);
					}

					if (!mtrlElem) {
						mtrlElem = xml.NewElement("material");
						root->InsertEndChild(mtrlElem);
					}

					mtrlElem->SetAttribute(attr->Name(), attr->Value());
				}
			}

			if (mtrlElem) {
				root->InsertEndChild(mtrlElem);
				MaterialLoader::onLoad(root);
			}
			else if (!path.empty()) {
				MaterialLoader::load(path);
			}
		}
	}

	void readObjects(
		const tinyxml2::XMLElement* root,
		std::vector<hitable*>& objs)
	{
		auto objRoot = root->FirstChildElement("objects");

		if (!objRoot) {
			return;
		}

		enum Type {
			Sphere,
			Cube,
			Object,
		};

		for (auto elem = objRoot->FirstChildElement("object"); elem != nullptr; elem = objRoot->NextSiblingElement("object")) {
			std::string path;
			std::string tag;

			vec3 trans(1);
			vec3 rotate(0);
			real scale(1);

			Type type = (Type)0;

			material* mtrl = nullptr;

			for (auto attr = elem->FirstAttribute(); attr != nullptr; attr = attr->Next()) {
				std::string attrName(attr->Name());

				if (attrName == "path") {
					path = attr->Value();
				}
				else if (attrName == "name") {
					tag = attr->Value();
				}
				else if (attrName == "type") {
					type = (Type)attr->UnsignedValue();
				}
				else if (attrName == "trans" || attrName == "rotate") {
					std::string text(attr->Value());

					std::vector<std::string> values;
					int num = split(text, values, ' ');

					vec3* v = (attrName == "trans" ? &trans : &rotate);

					for (int i = 0; i < std::min<int>(num, 3); i++) {
						(*v)[i] = (real)atof(values[i].c_str());
					}
				}
				else if (attrName == "scale") {
					scale = attr->DoubleValue();
				}
				else if (attrName == "material") {
					mtrl = AssetManager::getMtrl(attr->Value());
				}
			}

			object* obj = nullptr;

			if (type == Type::Object) {
				if (!tag.empty()) {
					obj = ObjLoader::load(tag, path);
				}
				else {
					obj = ObjLoader::load(path);
				}
			}

			mat4 mtxS;
			mtxS.asScale(scale);

			mat4 mtxRotX, mtxRotY, mtxRotZ;
			mtxRotX.asRotateByX(rotate.x);
			mtxRotY.asRotateByX(rotate.x);
			mtxRotZ.asRotateByX(rotate.x);

			mat4 mtxT;
			mtxT.asTrans(trans);

			auto mtxL2W = mtxT * mtxRotX * mtxRotY * mtxRotZ * mtxS;

			if (obj) {
				auto instance = new aten::instance<aten::object>(obj, mtxL2W);
				objs.push_back(instance);
			}
			else {
				if (type == Type::Cube) {
					auto cube = new aten::cube(1, 1, 1, mtrl);
					auto instance = new aten::instance<aten::cube>(cube, mtxL2W);
					objs.push_back(instance);
				}
				else {
					auto sphere = new aten::sphere(1, mtrl);
					auto instance = new aten::instance<aten::sphere>(sphere, mtxL2W);
					objs.push_back(instance);
				}
			}
		}
	}

	SceneLoader::SceneInfo SceneLoader::load(const std::string& path)
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

		std::vector<hitable*> objs;

		auto root = xml.FirstChildElement("scene");
		if (root) {
			readTextures(root);
			readMaterials(root);
			readObjects(root, objs);
		}

		SceneLoader::SceneInfo ret;

		return std::move(ret);
	}
#endif
}
