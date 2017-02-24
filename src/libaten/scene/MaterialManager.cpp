#include "scene/MaterialManager.h"

namespace aten {
	std::map<std::string, MaterialManager::MaterialCreator> MaterialManager::g_creators;
	std::map<std::string, material*> MaterialManager::g_mtrls;

	bool MaterialManager::addCreator(std::string tag, MaterialCreator creator)
	{
		auto it = g_creators.find(tag);

		if (it == g_creators.end()) {
			g_creators.insert(std::pair<std::string, MaterialCreator>(tag, creator));
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

	material* MaterialManager::get(std::string tag)
	{
		material* mtrl = nullptr;

		auto itMtrl = g_mtrls.find(tag);
		if (itMtrl != g_mtrls.end()) {
			mtrl = itMtrl->second;
		}
		else {
			auto itCreator = g_creators.find(tag);
			if (itCreator != g_creators.end()) {
				auto func = itCreator->second;
				if (func) {
					mtrl = func();
					g_mtrls.insert(std::pair<std::string, material*>(tag, mtrl));
				}
			}
		}

		return mtrl;
	}
}
