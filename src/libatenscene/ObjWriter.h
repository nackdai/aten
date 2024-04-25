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
            const std::vector<std::vector<int32_t>>& indices,
            FuncGetMtrlName func_get_mtrl_name);

        template <typename Mtrl>
        static bool writeMaterial(
            const aten::context& ctxt,
            const std::string& mtrlPath,
            const std::vector<Mtrl>& mtrls)
        {
            FILE* fp = fopen(mtrlPath.c_str(), "wt");

            for (const auto& mtrl : mtrls) {
                fprintf(fp, "newmtl %s\n", mtrl->name());

                const auto& param = mtrl->param();

                fprintf(fp, "Ns 1.000000\n");
                fprintf(fp, "Ka 0.000000 0.000000 0.000000\n");
                fprintf(fp, "Kd %.6f %.6f %.6f\n", param.baseColor.x, param.baseColor.y, param.baseColor.z);
                fprintf(fp, "Ks 0.000000 0.000000 0.000000\n");
                fprintf(fp, "Ni 1.000000\n");
                fprintf(fp, "d 1.000000\n");
                fprintf(fp, "illum 2\n");

                if (param.albedoMap >= 0) {
                    auto albedo = ctxt.GetTexture(param.albedoMap);
                    fprintf(fp, "map_Ka %s\n", albedo->name());
                    fprintf(fp, "map_Kd %s\n", albedo->name());
                }

                if (param.normalMap >= 0) {
                    auto normal = ctxt.GetTexture(param.normalMap);
                    fprintf(fp, "map_bump %s\n", normal->name());
                    fprintf(fp, "bump %s\n", normal->name());
                }

                fprintf(fp, "\n");
            }

            fclose(fp);

            return true;
        }

        static bool writeObjects(
            const std::string& path,
            const std::string& mtrlPath,
            const context& ctxt,
            const std::vector<std::shared_ptr<aten::PolygonObject>>& objs);

        bool runOnThread(
            std::function<void()> funcFinish,
            const std::string& path,
            const std::string& mtrlPath,
            const std::vector<aten::vertex>& vertices,
            const std::vector<std::vector<int32_t>>& indices,
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
