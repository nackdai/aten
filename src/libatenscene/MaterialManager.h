#pragma once

#include <map>
#include <functional>
#include "aten.h"

namespace aten {
	class MaterialManager {
	private:
		MaterialManager();
		~MaterialManager();

	public:
		using MaterialCreator = std::function<material*()>;

		static bool addCreator(std::string tag, MaterialCreator creator);
		static bool addMaterial(std::string tag, material* mtrl);

		static material* get(std::string tag);

	private:
		static std::map<std::string, MaterialCreator> g_creators;
		static std::map<std::string, material*> g_mtrls;
	};
}
