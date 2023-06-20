#pragma once

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

        FileInputStream(std::string_view path, std::string_view mode)
        {
            open(path, mode);
        }

    public:
        bool open(std::string_view path, std::string_view mode)
        {
            m_fp = fopen(path.data(), mode.data());
            return (m_fp != nullptr);
        }

        void close()
        {
            if (m_fp) {
                fclose(m_fp);
                m_fp = nullptr;
            }
        }

        uint32_t read(void* p, uint32_t size)
        {
            AT_ASSERT(m_fp);
            auto ret = fread(p, size, 1, m_fp);

            ret *= size;

            m_curPos = (uint32_t)ftell(m_fp);

            return (uint32_t)ret;
        }

        uint32_t curPos()
        {
            AT_ASSERT(m_fp);
            AT_ASSERT(m_curPos == (uint32_t)ftell(m_fp));
            return m_curPos;
        }

    private:
        FILE* m_fp{ nullptr };
        uint32_t m_curPos{ 0 };
    };
}

#define AT_STREAM_READ(in, p, size)    ((in)->read((p), (size)) == (size))
