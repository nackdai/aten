#pragma once

#include <vector>
#include "aten.h"

// TODO
// namespace で囲うと fatal error c1001 が発生する.
// 原因がわからないため、work around ではあるが namespace を使用しないようにする...

class ObjWriter {
public:
    ObjWriter() {}
    ~ObjWriter() {}

public:
    using FuncGetMtrlName = std::function<const char* (uint32_t)>;

    static bool write(
        const std::string& path,
        const std::string& mtrlPath,
        const std::vector<aten::vertex>& vertices,
        const std::vector<std::vector<int>>& indices,
        FuncGetMtrlName func_get_mtrl_name);

    static bool writeMaterial(
        const aten::context& ctxt,
        const std::string& mtrlPath,
        const std::vector<aten::material*>& mtrls);

    bool runOnThread(
        std::function<void()> funcFinish,
        const std::string& path,
        const std::string& mtrlPath,
        const std::vector<aten::vertex>& vertices,
        const std::vector<std::vector<int>>& indices,
        FuncGetMtrlName func_get_mtrl_name);

    void terminate();

    bool isRunningThread() const
    {
        return m_isRunning;
    }

private:
    aten::Thread m_thread;
    aten::Semaphore m_sema;

    std::atomic<bool> m_isRunning{ false };
    std::atomic<bool> m_isTerminate{ false };
};
