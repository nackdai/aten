#pragma once

#include "aten.h"

namespace aten
{
    class ObjLoader {
    private:
        ObjLoader() {}
        ~ObjLoader() {}

    public:
        static void setBasePath(const std::string& base);

        static object* load(
            const std::string& path,
            context& ctxt,
            bool needComputeNormalOntime = false);
        static object* load(
            const std::string& tag, 
            const std::string& path,
            context& ctxt,
            bool needComputeNormalOntime = false);

        static void load(
            std::vector<object*>& objs,
            const std::string& path,
            context& ctxt,
            bool willSeparate = false,
            bool needComputeNormalOntime = false);
        static void load(
            std::vector<object*>& objs,
            const std::string& tag, 
            const std::string& path,
            context& ctxt,
            bool willSeparate = false,
            bool needComputeNormalOntime = false);
    };
}
