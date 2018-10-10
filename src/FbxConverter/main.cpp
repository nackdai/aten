#include <cmdline.h>
#include <imgui.h>

#include "aten.h"
#include "atenscene.h"

#include "FbxImporter.h"
#include "MdlExporter.h"
#include "MtrlExporter.h"
#include "AnmExporter.h"

static const int WIDTH = 1280;
static const int HEIGHT = 720;

static const char* TITLE = "FbxConverter";

struct Options {
    std::string input;
    std::string output;

    std::string inputBasepath;
    std::string inputFilename;

    std::string anmBaseMdl;

    bool isExportModel{ true };
    bool isExportForGPUSkinning{ false };
};

bool parseOption(
    int argc, char* argv[],
    Options& opt)
{
    cmdline::parser cmd;

    {
        cmd.add<std::string>("input", 'i', "input filename", true);
        cmd.add<std::string>("output", 'o', "output filename base", false);
        cmd.add<std::string>("type", 't', "export type(m = model, a = animation)", true, "m");
        cmd.add<std::string>("base", 'b', "input filename for animation base model", false);
        cmd.add("gpu", 'g', "export for gpu skinning");

        cmd.add("help", '?', "print usage");
    }

    bool isCmdOk = cmd.parse(argc, argv);

    if (cmd.exist("help")) {
        std::cerr << cmd.usage();
        return false;
    }

    if (!isCmdOk) {
        std::cerr << cmd.error_full() << std::endl << cmd.usage();
        return false;
    }

    if (cmd.exist("input")) {
        opt.input = cmd.get<std::string>("input");

        std::string ext;

        aten::getStringsFromPath(
            opt.input,
            opt.inputBasepath,
            ext,
            opt.inputFilename);
    }
    else {
        std::cerr << cmd.error() << std::endl << cmd.usage();
        return false;
    }

    if (cmd.exist("output")) {
        opt.output = cmd.get<std::string>("output");
    }

    {
        auto type = cmd.get<std::string>("type");
        if (type == "m") {
            opt.isExportModel = true;
        }
        else if (type == "a") {
            opt.isExportModel = false;
        }
    }

    if (!opt.isExportModel) {
        // For animation.
        if (cmd.exist("base")) {
            opt.anmBaseMdl = cmd.get<std::string>("base");
        }
        else {
            std::cerr << cmd.error() << std::endl << cmd.usage();
            return false;
        }
    }

    opt.isExportForGPUSkinning = cmd.exist("gpu");

    return true;
}

int main(int argc, char* argv[])
{
#if 1
    Options opt;

    if (!parseOption(argc, argv, opt)) {
        return 0;
    }

    aten::SetCurrentDirectoryFromExe();

    if (opt.isExportModel)
    {
        aten::FbxImporter importer;

        importer.setIgnoreTexIdx(0);
        importer.open(opt.input.c_str());

        std::string output = opt.output;
        if (opt.output.empty()) {
            output = opt.inputFilename + (opt.isExportForGPUSkinning ? "_gpu" : "") + ".mdl";
        }

        MdlExporter::exportMdl(
            48, 
            output.c_str(), 
            &importer,
            opt.isExportForGPUSkinning);

        std::string mtrl;
        {
            output = opt.output;
            if (opt.output.empty()) {
                output = opt.inputFilename + ".mdl";
            }

            std::string basepath, ext, filename;

            aten::getStringsFromPath(
                output,
                basepath,
                ext,
                filename);

            mtrl = basepath + filename + "_mtrl" + ".xml";
        }

        MtrlExporter::exportMaterial(mtrl.c_str(), &importer);

        // Fbx viewer don't support GPU skinning format data.
        if (opt.isExportForGPUSkinning) {
            return 1;
        }
    }
    else {
        aten::FbxImporter importer;
        
        importer.open(opt.input.c_str(), true);
        importer.readBaseModel(opt.anmBaseMdl.c_str());

        std::string output = opt.output;
        if (output.empty()) {
            output = opt.inputFilename + ".anm";
        }

        AnmExporter::exportAnm(output.c_str(), 0, &importer);
    }
#else
    aten::SetCurrentDirectoryFromExe();

#if 0
    {
        aten::FbxImporter importer;

        importer.setIgnoreTexIdx(0);
        importer.open("../../asset/unitychan/unitychan.fbx");
        MdlExporter::exportMdl(48, "unitychan.mdl", &importer);
        MtrlExporter::exportMaterial("unitychan_mtrl.xml", &importer);
    }
#endif

#if 0
    {
        aten::FbxImporter importer;
        importer.open("../../asset/unitychan/unitychan_WAIT01.fbx", true);
        importer.readBaseModel("../../asset/unitychan/unitychan.fbx");
        AnmExporter::exportAnm("unitychan.anm", 0, &importer);
    }
#endif
#endif

    return 1;
}
