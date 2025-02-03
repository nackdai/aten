#include "aten.h"

#include <cmdline.h>
#include <imgui.h>

#include "../common/app_base.h"

constexpr int32_t WIDTH = 1280;
constexpr int32_t HEIGHT = 720;

class ImageViewerApp : public App {
public:
    static constexpr const char* TITLE = "ImageViewer";

    ImageViewerApp(int32_t width, int32_t height) : App(width, height) {}
    ~ImageViewerApp() = default;

    ImageViewerApp(const ImageViewerApp&) = delete;
    ImageViewerApp(ImageViewerApp&&) = delete;
    ImageViewerApp operator=(const ImageViewerApp&) = delete;
    ImageViewerApp operator=(ImageViewerApp&&) = delete;

    bool Init()
    {
        return true;
    }

    bool Run()
    {
        if (image_) {
            if (!image_->initAsGLTexture()) {
                AT_ASSERT(false);
                return false;
            }
        }

        if (!blitter_.IsValid()) {
            visualizer_ = aten::visualizer::init(args_.width, args_.height);

            if (!blitter_.init(
                args_.width, args_.height,
                "../shader/fullscreen_vs.glsl",
                "../shader/fullscreen_fs.glsl"))
            {
                AT_ASSERT(false);
                return false;
            }

            visualizer_->addPostProc(&blitter_);
        }

        aten::RasterizeRenderer::clearBuffer(
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
            aten::vec4(0, 0.5f, 1.0f, 1.0f),
            1.0f,
            0);

        if (image_) {
            visualizer_->renderGLTexture(image_.get(), false);
        }

        if (ImGui::Checkbox("Read As sRGB", &need_read_as_srgb_)) {
            if (image_ && !image_file_path_.empty()) {
                // Re-read.
                ReadImage(image_file_path_);
            }
        }

        return true;
    }

    void OnClose()
    {
    }

    void OnDrop(std::string_view path)
    {
        ctxt_.CleanAll();

        std::ignore = aten::Image::Load("", path, ctxt_);
    }

    void OnDropFile(std::string_view path)
    {
        image_file_path_ = path;
        ReadImage(path);
    }

    bool ParseArgs(int32_t argc, char* argv[])
    {
        cmdline::parser cmd;
        {
            cmd.add<int32_t>("width", 'w', "texture width", false);
            cmd.add<int32_t>("height", 'h', "texture height", false);

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

        if (cmd.exist("height")) {
            args_.width = cmd.get<int32_t>("height");
        }

        if (cmd.exist("width")) {
            args_.width = cmd.get<int32_t>("width");
        }

        return true;
    }

    int32_t width() const
    {
        return args_.width;
    }

    int32_t height() const
    {
        return args_.height;
    }

    aten::context& GetContext()
    {
        return ctxt_;
    }

private:
    void ReadImage(std::string_view path)
    {
        const AT_NAME::sRGBGammaEncoder srgb_encoder;
        const AT_NAME::LinearEncoder linear_encoder;

        image_ = aten::Image::Load(
            "", path,
            ctxt_,
            need_read_as_srgb_ ? static_cast<const AT_NAME::ColorEncoder*>(&srgb_encoder) : &linear_encoder);
    }

    struct Args {
        int32_t width{ 1280 };
        int32_t height{ 720 };
    } args_;

    aten::context ctxt_;
    std::shared_ptr<aten::texture> image_;

    std::string image_file_path_;

    std::shared_ptr<aten::visualizer> visualizer_;

    aten::Blitter blitter_;

    bool need_read_as_srgb_{ false };
};

int32_t main(int32_t argc, char* argv[])
{
    aten::SetCurrentDirectoryFromExe();

    auto app = std::make_shared<ImageViewerApp>(WIDTH, HEIGHT);

    if (!app->Init()) {
        AT_ASSERT(false);
        return 1;
    }

    auto wnd = std::make_shared<aten::window>();

    aten::window::MesageHandlers handlers;
    handlers.OnRun = [&app]() { return app->Run(); };
    handlers.OnClose = [&app]() { app->OnClose();  };
    handlers.OnDropFile = [&app](std::string_view path) { app->OnDropFile(path);  };

    auto id = wnd->Create(
        app->width(), app->height(),
        ImageViewerApp::TITLE,
        handlers);

    if (id >= 0) {
        app->GetContext().SetIsWindowInitialized(true);
    }
    else {
        AT_ASSERT(false);
        return 1;
    }

    wnd->Run();

    app.reset();

    wnd->Terminate();
}
