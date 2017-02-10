#pragma once

#include <string>
#include "defs.h"

namespace aten {
	class texture;

	class ImageLoader {
	private:
		ImageLoader() {}
		~ImageLoader() {}

	public:
		static texture* load(const std::string& path);
	};
}