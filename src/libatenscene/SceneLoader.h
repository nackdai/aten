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

		struct ProcInfo {
			std::string type;
			Values values;
		};

		struct SceneInfo {
			scene* scene{ nullptr };
			aten::Destination dst;
			std::string rendererType;
			std::vector<ProcInfo> preprocs;
			std::vector<ProcInfo> postprocs;
		};

		static SceneInfo load(const std::string& path);
	};
}
