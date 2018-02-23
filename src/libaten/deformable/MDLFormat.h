#pragma once

#include "defs.h"
#include "types.h"

#define AT_CC4(c0, c1, c2, c3)	(((c0) << 24) | ((c1) << 16) | ((c2) << 8) | (c3))

namespace aten
{
	enum class MdlChunkMagic : uint32_t {
		// TODO
		Mesh = AT_CC4('m', 'e', 's', 'h'),
		Joint = AT_CC4('s', 'k', 'l', 't'),
		Terminate = AT_CC4('f', 'e', 'n', 'd'),
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
