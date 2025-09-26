#pragma once

#include <string>
#include <algorithm>
#include <filesystem>

namespace aten {
    inline void getStringsFromPath(
        std::string_view path,
        std::string& pathname,
        std::string& extname,
        std::string& filename)
    {
        std::filesystem::path p = path;

        pathname = p.parent_path().generic_string();
        extname = p.extension().generic_string();
        filename = p.stem().generic_string();

        if (path == pathname) {
            pathname = "./";
        }
    }

    inline std::string removeTailPathSeparator(std::string_view path)
    {
        std::string ret(path);

        // Remove tail '\' or '/'.

        auto len = ret.length();
        auto ch = ret[len - 1];

        if (ch == '\\' || ch == '/') {
            ret.pop_back();
        }

        return ret;
    }

    inline uint32_t split(std::string_view txt, std::vector<std::string>& strs, char ch)
    {
        auto pos = txt.find(ch);
        size_t initialPos = 0;
        strs.clear();

        auto s = txt.size();

        // Decompose statement
        //while (pos != std::string::npos) {
        while (pos <= s) {
            strs.push_back(std::string(txt.substr(initialPos, pos - initialPos + 1)));
            initialPos = pos + 1;

            pos = txt.find(ch, initialPos);
        }

        // Add the last one
        strs.push_back(std::string(txt.substr(initialPos, std::min<size_t>(pos, txt.size()) - initialPos + 1)));

        return (uint32_t)strs.size();
    }
}
