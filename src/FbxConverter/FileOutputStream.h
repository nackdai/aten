#pragma once

#include "defs.h"
#include "types.h"

enum FileIoMode {
	Bin,
	Text,
};

enum FilSeekPos {
	Head,
	Tail,
	Cur,
};

class FileOutputStream
{
public:
	FileOutputStream()
    {
    }
    ~FileOutputStream()
    {
        finalize();
    }

	FileOutputStream(const FileOutputStream& rhs) = delete;
	FileOutputStream& operator=(const FileOutputStream& rhs) = delete;

public:
    bool open(const char* path)
    {
        return open(path, FileIoMode::Bin);
    }

    bool open(const char* path, FileIoMode mode)
    {
        bool ret = false;

        if (m_File != nullptr) {
            AT_ASSERT(false);
            finalize();
        }

		m_File = fopen(path, FileIoMode::Text ? "wt" : "wb");
        ret = (m_File != nullptr);

        return ret;
    }

    void finalize()
    {
		if (m_File) {
			fclose(m_File);
		}

        m_File = nullptr;
        m_Pos = 0;
    }

    // 出力
    uint32_t write(const void* buf, uint32_t offset, size_t size)
    {
        AT_ASSERT(m_File != nullptr);
        AT_ASSERT(size > 0);

        if (offset > 0) {
            seek(offset, FilSeekPos::Cur);
        }

        uint32_t ret = (uint32_t)fwrite(buf, size, 1, m_File);
        AT_ASSERT(ret == size);

        if (ret == size) {
            m_Pos += offset + (uint32_t)size;
        }

        return ret;
    }

    bool write(const char* text)
    {
        AT_ASSERT(m_File != nullptr);

        int result = fprintf(m_File, "%s", text);
        AT_ASSERT(result >= 0);

        return (result >= 0);
    }

    // サイズ取得
    uint32_t getSize()
    {
        AT_ASSERT(m_File != nullptr);

        uint32_t ret = (uint32_t)ftell(m_File);
        return ret;
    }

    // 現在位置取得
    uint32_t getCurPos()
    {
        return m_Pos;
    }

    // シーク
    bool seek(int32_t offset, FilSeekPos seekPos)
    {
        AT_ASSERT(m_File != nullptr);

        uint32_t nPos = SEEK_SET;

        switch (seekPos) {
		case FilSeekPos::Head:
            nPos = SEEK_SET;
            break;
		case FilSeekPos::Cur:
            nPos = SEEK_CUR;
            break;
		case FilSeekPos::Tail:
            // 出力時にはファイル終端は存在しないので・・・
			AT_VRETURN_FALSE(false);
			break;
        }

        bool ret = true;
        {
            // シーク
            ret = (fseek(m_File, offset, nPos) == 0);

            if (ret) {
                // 現在位置更新
                switch (seekPos) {
				case FilSeekPos::Head:
                    m_Pos = offset;
                    break;
				case FilSeekPos::Cur:
                    m_Pos += offset;
                    break;
				case FilSeekPos::Tail:
                    AT_VRETURN_FALSE(false);
                    break;
                }
            }
        }
        AT_ASSERT(ret);

        return ret;
    }

    bool isValid()
    {
        return (m_File != NULL);
    }

private:
	FILE* m_File{ nullptr };
	uint32_t m_Pos{ 0 };
};

#define OUTPUT_WRITE(out, p, offset, size)           ((out)->write((p), (offset), (size)) == (size))
#define OUTPUT_WRITE_VRETURN(out, p, offset, size)   AT_VRETURN(OUTPUT_WRITE(out, p, offset, size), false)

class IoStreamSeekHelper {
protected:
	IoStreamSeekHelper() {}

public:
	IoStreamSeekHelper(FileOutputStream* pOut)
	{
		m_pOut = pOut;
		m_nPos = 0;
		m_nAnchorPos = 0;
	}

	~IoStreamSeekHelper() {}

public:
	bool skip(uint32_t nSkip)
	{
		m_nPos = m_pOut->getCurPos();
		AT_VRETURN_FALSE(m_pOut->seek(nSkip, FilSeekPos::Cur));
		return true;
	}

	void step(uint32_t nStep)
	{
		m_nPos += nStep;
	}

	bool returnTo()
	{
		uint32_t nCurPos = m_pOut->getCurPos();
		int32_t nOffset = m_nPos - nCurPos;
		AT_VRETURN_FALSE(m_pOut->seek(nOffset, FilSeekPos::Cur));
		return true;
	}

	bool returnWithAnchor()
	{
		m_nAnchorPos = m_pOut->getCurPos();
		int32_t nOffset = m_nPos - m_nAnchorPos;
		AT_VRETURN_FALSE(m_pOut->seek(nOffset, FilSeekPos::Cur));
		return true;
	}

	bool returnToAnchor()
	{
		uint32_t nCurPos = m_pOut->getCurPos();
		int32_t nOffset = m_nAnchorPos - nCurPos;
		AT_VRETURN_FALSE(m_pOut->seek(nOffset, FilSeekPos::Cur));
		return true;
	}

	FileOutputStream* getOutputStream() { return m_pOut; }

protected:
	FileOutputStream* m_pOut;
	uint32_t m_nPos;
	uint32_t m_nAnchorPos;
};