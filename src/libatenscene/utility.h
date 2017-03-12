#pragma once

#include <string>
#include <algorithm>

namespace aten {
	inline void getStringsFromPath(
		const std::string& path,
		std::string& pathname,
		std::string& extname,
		std::string& filename)
	{
		std::string filepath = path;

		// Get tag from file path.
		// Replace \\->/
		std::replace(filepath.begin(), filepath.end(), '\\', '/');

		int pathPos = (int)filepath.find_last_of("/") + 1;
		int extPos = (int)filepath.find_last_of(".");

		pathname = filepath.substr(0, pathPos + 1);
		extname = filepath.substr(extPos, filepath.size() - extPos);
		filename = filepath.substr(pathPos, extPos - pathPos);
	}

	inline std::string removeTailPathSeparator(const std::string path)
	{
		std::string ret = path;

		// Remove tail '\' or '/'.

		auto len = ret.length();
		auto ch = ret[len - 1];

		if (ch == '\\' || ch == '/') {
			ret.pop_back();
		}

		return std::move(ret);
	}
}