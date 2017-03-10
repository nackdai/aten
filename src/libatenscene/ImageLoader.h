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
		static void setBasePath(const std::string& base);

		static texture* load(const std::string& path);

		static bool add(const std::string& tag, texture* tex);
		static texture* get(const std::string& tag);
	};
}