#include <Shlwapi.h>
#include "os/system.h"

namespace aten
{
    bool SetCurrentDirectoryFromExe()
    {
        static char buf[_MAX_PATH];

        // 実行プログラムのフルパスを取得
        {
            DWORD result = ::GetModuleFileName(
                NULL,
                buf,
                sizeof(buf));
            AT_ASSERT(result > 0);
        }

        // ファイル名を取り除く
        auto result = ::PathRemoveFileSpec(buf);
        AT_ASSERT(result);

        // カレントディレクトリを設定
        result = ::SetCurrentDirectory(buf);
        AT_ASSERT(result);

        return result ? true : false;
    }
}
