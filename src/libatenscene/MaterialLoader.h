#pragma once

#include "aten.h"

namespace aten {
	class MaterialLoader {
	private:
		MaterialLoader();
		~MaterialLoader();

	public:
		using MaterialCreator = std::function<material*(Values&)>;

		static void setBasePath(const std::string& base);

		static bool addCreator(std::string type, MaterialCreator creator);

		static bool load(const std::string& path);

#ifdef USE_JSON
		static void onLoad(const std::string& strJson);
#else
		static void onLoad(const void* xmlRoot);
#endif

		static material* create(std::string type, Values& values);
	};
}
