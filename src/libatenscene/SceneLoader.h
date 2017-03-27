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

		static scene* load(const std::string& path);
	};
}
