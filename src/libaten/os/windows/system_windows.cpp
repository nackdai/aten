#include <unistd.h>
#include <string.h>
#include "os/system.h"

namespace aten
{
	static uint64_t ptrToInt(void* p)
	{
		AT_STATICASSERT(sizeof(uint64_t) >= sizeof(void*));

		union {
			uint64_t val;
			void* ptr;
		} conv;

		conv.ptr = p;
		return conv.val;
	}

	static inline uint64_t getPtrDistance(void* p0, void* p1)
	{
		uint64_t pos_0 = ptrToInt(p0);
		uint64_t pos_1 = ptrToInt(p1);

		uint64_t _max = std::max(pos_0, pos_1);
		uint64_t _min = std::min(pos_0, pos_1);

		return (_max - _min);
	}

	static inline int32_t getExecuteFilePath(char* path, size_t pathBufSize)
	{
		// 実行プログラムのフルパスを取得
		int32_t result = readlink("/proc/self/exe", path, pathBufSize);
		AT_VRETURN(result > 0, 0);

		return result;
	}

	bool SetCurrentDirectoryFromExe()
	{
		static char path[256];

		// 実行プログラムのフルパスを取得
		int pathSize = getExecuteFilePath(path, sizeof(path));
		AT_VRETURN(pathSize > 0);

		char* tmp = const_cast<char*>(path);

		// ファイル名を取り除く
		char* p = ::strrchr(tmp, '/');
		AT_VRETURN(p != nullptr);

		auto diff = getPtrDistance(tmp, p);

		tmp[diff] = 0;
		p = tmp;

		// カレントディレクトリを設定
		int result = ::chdir(p);
		AT_ASSERT(result >= 0);

		return result >= 0;
	}
}
