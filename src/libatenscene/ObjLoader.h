#pragma once

#include "aten.h"

namespace aten
{
	class object;

	class ObjLoader {
	private:
		ObjLoader() {}
		~ObjLoader() {}

	public:
		static object* load(const std::string& path);
		static object* load(const std::string& tag, const std::string& path);
	};
}
