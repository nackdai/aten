#include <cmdline.h>

#include "aten.h"
#include "atenscene.h"

struct Options {
    std::string input;
    std::string output;
};

bool parseOption(
    int32_t argc, char* argv[],
    cmdline::parser& cmd,
    Options& opt)
{
    {
        cmd.add<std::string>("input", 'i', "input filename", true);
        cmd.add<std::string>("output", 'o', "output filename base", false);

        cmd.add("help", '?', "print usage");
    }

    bool isCmdOk = cmd.parse(argc, argv);

    if (cmd.exist("help")) {
        std::cerr << cmd.usage();
        return false;
    }

    if (!isCmdOk) {
        std::cerr << cmd.error() << std::endl << cmd.usage();
        return false;
    }

    if (cmd.exist("input")) {
        opt.input = cmd.get<std::string>("input");
    }
    else {
        std::cerr << cmd.error() << std::endl << cmd.usage();
        return false;
    }

    if (cmd.exist("output")) {
        opt.output = cmd.get<std::string>("output");
    }
    else {
        std::string pathname;
        std::string filename;
        std::string extname;

        aten::getStringsFromPath(opt.input, pathname, extname, filename);

        opt.output = filename + ".sbvh";
    }

    return true;
}


int32_t main(int32_t argc, char* argv[])
{
    Options opt;
    cmdline::parser cmd;

    if (!parseOption(argc, argv, cmd, opt)) {
        return 0;
    }

    aten::SetCurrentDirectoryFromExe();

    aten::context ctxt;

    auto objs = aten::ObjLoader::load(opt.input, ctxt,
        [&](std::string_view name, aten::context& ctxt,
            aten::MaterialType type, const aten::vec3& mtrl_clr,
            const std::string& albedo, const std::string& nml) -> auto {
        (void)albedo;
        (void)nml;

        aten::MaterialParameter param;
        param.type = type;
        param.baseColor = mtrl_clr;

        auto mtrl = ctxt.CreateMaterialWithMaterialParameter(
            name,
            param,
            nullptr, nullptr, nullptr);

        return mtrl;
    },
        true, true);


    if (objs.empty()) {
        // TODO
        return 0;
    }

    // Not use.
    // But specify internal accelerator type in constructor...
    aten::AcceleratedScene<aten::sbvh> scene;

    try {
        static char buf[2048] = { 0 };

        for (int32_t i = 0; i < objs.size(); i++) {
            auto& obj = objs[i];

            std::string output;

            if (i == 0) {
                output = opt.output;
            }
            else {
                // TODO
                sprintf(buf, "%d_%s\0", i, opt.output.c_str());
                output = std::string(buf);
            }

            obj->exportInternalAccelTree(ctxt, output.c_str());
        }
    }
    catch (const std::exception& e) {
        // TODO

        return 0;
    }

    return 1;
}
