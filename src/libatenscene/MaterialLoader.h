#pragma once

#include <map>
#include <functional>
#include "aten.h"

namespace aten {
	class MaterialLoader {
	private:
		MaterialLoader();
		~MaterialLoader();

	public:
		using MaterialCreator = std::function<material*(Values&)>;

		static bool addCreator(std::string type, MaterialCreator creator);
		static bool addMaterial(std::string tag, material* mtrl);

		static material* load(std::string path);
		static material* load(std::string tag, std::string path);

		static material* get(std::string tag);

	private:
		static material* create(std::string type, Values& values);
	};
}
