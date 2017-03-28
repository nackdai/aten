#include "AssetManager.h"

namespace aten {
	union Asset {
		texture* tex{ nullptr };
		material* mtrl;
		object* obj;

		Asset() {}
		Asset(texture* t) : tex(t) {}
		Asset(material* m) : mtrl(m) {}
		Asset(object* o) : obj(o) {}
	};

	using AssetStorage = std::map<std::string, Asset>;
	static AssetStorage g_assets[AssetManager::AssetType::Num];

	static const char* AssetTypeName[AssetManager::AssetType::Num] = {
		"Texture",
		"Material",
		"Object",
	};

	static bool registerAsset(
		const std::string& name,
		Asset& asset,
		AssetManager::AssetType type)
	{
		auto& mapAsset = g_assets[type];

		auto it = mapAsset.find(name);
		if (it != mapAsset.end()) {
			AT_PRINTF("Registered already [%s] (%s)\n", name.c_str(), AssetTypeName[type]);
			return false;
		}

		mapAsset.insert(std::pair<std::string, Asset>(name, asset));

		return true;
	}

	static Asset getAsset(
		const std::string& name,
		AssetManager::AssetType type)
	{
		auto& mapAsset = g_assets[type];

		auto it = mapAsset.find(name);

		if (it == mapAsset.end()) {
			//AT_ASSERT(false);
			AT_PRINTF("Asset is not registered [%s] (%s)\n", name.c_str(), AssetTypeName[type]);
			return Asset();
		}

		auto& asset = it->second;

		return asset;
	}

	bool AssetManager::registerMtrl(const std::string& name, material* mtrl)
	{
		return registerAsset(name, Asset(mtrl), AssetType::Material);
	}

	material* AssetManager::getMtrl(const std::string& name)
	{
		auto& asset = getAsset(name, AssetType::Material);
		return asset.mtrl;
	}

	bool AssetManager::registerTex(const std::string& name, texture* tex)
	{
		return registerAsset(name, Asset(tex), AssetType::Texture);
	}

	texture* AssetManager::getTex(const std::string& name)
	{
		auto& asset = getAsset(name, AssetType::Texture);
		return asset.tex;
	}

	bool AssetManager::registerObj(const std::string& name, object* obj)
	{
		return registerAsset(name, Asset(obj), AssetType::Object);
	}

	object* AssetManager::getObj(const std::string& name)
	{
		auto& asset = getAsset(name, AssetType::Object);
		return asset.obj;
	}
}
