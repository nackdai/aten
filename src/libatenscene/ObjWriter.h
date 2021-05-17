#pragma once

#include <vector>
#include "aten.h"

namespace aten {
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

        static bool writeObjects(
            const std::string& path,
            const std::string& mtrlPath,
            const context& ctxt,
            const std::vector<std::shared_ptr<aten::object>>& objs);

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
}
