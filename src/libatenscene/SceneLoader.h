#pragma once

#include "aten.h"

namespace aten
{
	class SceneLoader {
	private:
		SceneLoader() {}
		~SceneLoader() {}

	public:
		static void setBasePath(const std::string& base);

		struct SceneInfo {
			scene* scene{ nullptr };
			aten::Destination dst;
			std::string rendererType;
			std::vector<std::string> preprocs;
			std::vector<std::string> postprocs;
		};

		static SceneInfo load(const std::string& path);
	};
}
