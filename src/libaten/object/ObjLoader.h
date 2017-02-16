#pragma once

#include "types.h"

namespace aten
{
	class object;

	class ObjLoader {
	private:
		ObjLoader() {}
		~ObjLoader() {}

	public:
		static object* load(const char* path);
	};
}
