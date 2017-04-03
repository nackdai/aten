#pragma once

#include "aten.h"

namespace aten {
	class MaterialLoader {
		friend class SceneLoader;

	private:
		MaterialLoader();
		~MaterialLoader();

	public:
		using MaterialCreator = std::function<material*(Values&)>;

		static void setBasePath(const std::string& base);

		static bool addCreator(std::string type, MaterialCreator creator);

		static void load(const std::string& path);

	private:
#ifdef USE_JSON
		static void onLoad(const std::string& strJson);
#else

#endif

		static material* create(std::string type, Values& values);
	};
}
