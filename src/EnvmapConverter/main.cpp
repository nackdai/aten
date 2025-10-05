#include "aten.h"
#include "atenscene.h"

#include <cmdline.h>
#include <imgui.h>

#include "envmap.h"

class EnvmapConvereterApp {
public:
    static constexpr const char* TITLE = "EnvmapConverter";

    EnvmapConvereterApp() = default;
    ~EnvmapConvereterApp() = default;

    EnvmapConvereterApp(const EnvmapConvereterApp&) = delete;
    EnvmapConvereterApp(EnvmapConvereterApp&&) = delete;
    EnvmapConvereterApp operator=(const EnvmapConvereterApp&) = delete;
    EnvmapConvereterApp operator=(EnvmapConvereterApp&&) = delete;

    bool Run()
    {
        src_ = EnvMap::LoadEnvmap(
            ctxt_,
            args_.in_type,
            args_.input);
        if (!src_) {
            std::cerr << "failed to load env map: " << args_.input << std::endl;
            return false;
        }

        dst_ = EnvMap::CreateEmptyEnvmap(
            args_.out_type,
            args_.width, args_.height);

        if (!dst_) {
            std::cerr << "failed to create empty env map" << std::endl;
            return false;
        }

        EnvMap::Convert(src_, dst_);

        dst_->SaveAsPng(args_.output);

        return true;
    }

    EnvMapType ConvertStringToEnvMapType(const std::string& type_str)
    {
        if (type_str == "equirect") {
            return EnvMapType::Equirect;
        }
        else if (type_str == "cube") {
            return EnvMapType::CubeMap;
        }
        else if (type_str == "mirror") {
            return EnvMapType::Mirror;
        }
        else if (type_str == "angular") {
            return EnvMapType::Angular;
        }
        else {
            std::cerr << "unknown env map type: " << type_str << std::endl;
            return EnvMapType::Invalid;
        }
    }

    bool ParseArgs(int32_t argc, const char** argv)
    {
        cmdline::parser cmd;
        {
            cmd.add<std::string>("input", 'i', "input env map", true);
            cmd.add<std::string>("in_type", 0, "input env type [equirect, cube, mirror, anglar]", true);
            cmd.add<std::string>("output", 'o', "output env map(png)", false, "result");
            cmd.add<std::string>("out_type", 0, "output env type [equirect, cube, mirror, anglar]", true);
            cmd.add<int32_t>("width", 'w', "output texture width", false);
            cmd.add<int32_t>("height", 'h', "output texture height", false);
            cmd.add<std::string>("help", '?', "print usage", false);
        }

        bool is_cmd_valid = cmd.parse(argc, argv);

        if (cmd.exist("help")) {
            std::cerr << cmd.usage();
            return false;
        }

        if (!is_cmd_valid) {
            std::cerr << cmd.error() << std::endl << cmd.usage();
            return false;
        }

        args_.input = cmd.get<std::string>("input");

        if (cmd.exist("output")) {
            args_.output = cmd.get<std::string>("output");
        }
        else {
            args_.output = "result.png";
        }

        auto in_type_str = cmd.get<std::string>("in_type");
        args_.in_type = ConvertStringToEnvMapType(in_type_str);
        if (args_.in_type == EnvMapType::Invalid) {
            return false;
        }

        auto out_type_str = cmd.get<std::string>("out_type");
        args_.out_type = ConvertStringToEnvMapType(out_type_str);
        if (args_.in_type == EnvMapType::Invalid) {
            return false;
        }

        if (cmd.exist("width")) {
            args_.width = cmd.get<int32_t>("width");
        }
        else {
            if (args_.out_type == EnvMapType::Equirect) {
                args_.width = 1024;
            }
            else {
                args_.width = 512;
            }
        }

        if (cmd.exist("height")) {
            args_.height = cmd.get<int32_t>("height");
        }
        else {
            if (args_.out_type == EnvMapType::Equirect) {
                args_.height = args_.width / 2;
            }
            else {
                args_.height = args_.width;
            }
        }

        return true;
    }

private:
    struct Args {
        std::string input;
        EnvMapType in_type{ EnvMapType::Equirect };
        std::string output;
        EnvMapType out_type{ EnvMapType::CubeMap };
        int32_t width{ -1 };
        int32_t height{ -1 };
    } args_;

    aten::context ctxt_;
    std::shared_ptr<EnvMap> src_;
    std::shared_ptr<EnvMap> dst_;
};

int32_t main(int32_t argc, char* argv[])
{
    aten::SetCurrentDirectoryFromExe();

    auto app = std::make_shared<EnvmapConvereterApp>();

#if 1
    if (!app->ParseArgs(argc, const_cast<const char**>(argv))) {
        AT_ASSERT(false);
        return 1;
    }

    if (!app->Run()) {
        AT_ASSERT(false);
        return 1;
    }
#elif 0
    // for debug
    std::array args = {
        "EnvmapConverter",
        "-i", "sphere_map_sample.png",
        "--in_type", "equirect",
        "-o", "result.png",
        "--out_type", "cube",
        "-w", "512",
        "-h", "512",
    };
    if (!app->ParseArgs(args.size(), args.data())) {
        AT_ASSERT(false);
        return 1;
    }
    if (!app->Run()) {
        AT_ASSERT(false);
        return 1;
    }
#else
    // The equirect image is the master.
    // And, we convert from the equirect image to other types.
    // So, we have to list the equirect image first.
    // There is not master for other types. The images are generated from the equirect image on the fly.
    // So, the listed image files names are based on the equirect.
    constexpr std::array datas = {
        std::make_tuple(EnvMapType::Equirect, "equirect", "sphere_map_sample.png"),
        std::make_tuple(EnvMapType::CubeMap, "cube", "cube_from_equirect.png"),
        std::make_tuple(EnvMapType::Mirror, "mirror", "mirror_from_equirect.png"),
        std::make_tuple(EnvMapType::Angular, "angular", "angular_from_equirect.png")
    };

    std::array args = {
        "EnvmapConverter",
        "-i", "sphere_map_sample.png",
        "--in_type", "equirect",
        "-o", "result.png",
        "--out_type", "cube",
        "-w", "512",
        "-h", "512",
    };

    for (size_t i = 0; i < datas.size(); i++) {
        const auto& in_data = datas[i];

        const auto in_type_as_str = std::get<1>(in_data);

        for (size_t n = 0; n < datas.size(); n++) {
            if (n == i) {
                continue;
            }

            const auto& out_data = datas[n];

            const auto out_type = std::get<0>(out_data);

            // Input.
            args[2] = std::get<2>(in_data);
            args[4] = in_type_as_str;

            // Output.
            args[8] = std::get<1>(out_data);;

            std::string output(std::get<1>(out_data));
            output += "_from_";
            output += in_type_as_str;
            output += ".png";

            if (out_type == EnvMapType::Equirect) {
                // width, height
                args[10] = "1000";
                args[12] = "500";
            }
            else {
                // width, height
                args[10] = "512";
                args[12] = "512";
            }

            args[6] = output.data();

            if (!app->ParseArgs(args.size(), args.data())) {
                AT_ASSERT(false);
                return 1;
            }
            if (!app->Run()) {
                AT_ASSERT(false);
                return 1;
            }
        }
    }
#endif

    return 0;
}
