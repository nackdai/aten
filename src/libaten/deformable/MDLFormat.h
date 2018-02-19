#pragma once

#include "defs.h"
#include "types.h"

namespace aten
{
	enum class MdlChunkMagic : uint32_t {
		// TODO
		Mesh = 0x01234567,
		Joint = 0x12345678,
		Terminate = 0x7fffffff,
	};

	// NOTE
	// フォーマット
	// +--------------------+
	// |   ファイルヘッダ   |
	// +--------------------+
	// |  メッシュチャンク  |
	// +--------------------+
	// | スケルトンチャンク |
	// +--------------------+

	struct MdlHeader {
		uint32_t magic{ 0 };
		uint32_t version{ 0 };

		uint32_t sizeHeader{ 0 };
		uint32_t sizeFile{ 0 };
	};

	struct MdlChunkHeader {
		MdlChunkMagic magicChunk;
	};
}
