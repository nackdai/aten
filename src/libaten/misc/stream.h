#pragma once

#include <iostream>
#include <fstream>
#include <string>

#include "defs.h"
#include "types.h"

namespace aten
{
    class FileInputStream {
    public:
        FileInputStream() = default;
        ~FileInputStream()
        {
            close();
        }

        FileInputStream(std::string_view path)
        {
            open(path);
        }

    public:
        bool open(std::string_view path)
        {
            ifs_.open(path.data(), std::ios_base::in | std::ios_base::binary);
            bool is_failed = !ifs_;
            return !is_failed;
        }

        void close()
        {
            if (ifs_) {
                ifs_.close();
            }
        }

        size_t read(void* p, size_t size)
        {
            AT_ASSERT(ifs_);
            const auto before_read_pos = ifs_.tellg();
            ifs_.read(reinterpret_cast<char*>(p), size);
            const auto after_read_pos = ifs_.tellg();
            const auto read_size = after_read_pos - before_read_pos;

            return read_size;
        }

        size_t tell()
        {
            AT_ASSERT(ifs_);
            const auto pos = ifs_.tellg();
            return pos;
        }

    private:
        std::ifstream ifs_;
    };
}

#define AT_STREAM_READ(in, p, size)    ((in)->read((p), (size)) == (size))
