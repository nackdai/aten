#pragma once

#include <map>
#include <functional>
#include "aten.h"

namespace aten {
	class AssetManager {
	private:
		AssetManager() {}
		~AssetManager() {}

	public:
		enum AssetType {
			Texture,
			Material,
			Object,

			Num,
		};

		static bool registerMtrl(const std::string& name, material* mtrl);
		static material* getMtrl(const std::string& name);

		static bool registerTex(const std::string& name, texture* tex);
		static texture* getTex(const std::string& name);

		static bool registerObj(const std::string& name, object* obj);
		static object* getObj(const std::string& name);

		static void suppressWarnings();
	};
}
