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

		static void load(const std::string& path);

	private:
		static void onLoad(const std::string& strJson);
		static material* create(std::string type, Values& values);
	};
}
