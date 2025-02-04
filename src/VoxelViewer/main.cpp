#include <cmdline.h>
#include <imgui.h>

#include "aten.h"
#include "atenscene.h"
#include "VoxelViewer.h"

constexpr int32_t WIDTH = 1280;
constexpr int32_t HEIGHT = 720;
constexpr const char* TITLE = "VoxelViewer";

class VoxelViewerApp {
public:
    VoxelViewerApp() = default;
    ~VoxelViewerApp() = default;

    VoxelViewerApp(const VoxelViewerApp&) = delete;
    VoxelViewerApp(VoxelViewerApp&&) = delete;
    VoxelViewerApp operator=(const VoxelViewerApp&) = delete;
    VoxelViewerApp operator=(VoxelViewerApp&&) = delete;

    bool Init(int32_t argc, char* argv[])
    {
        // TODO
        //args_.input = "../../asset/cornellbox/orig.obj";
        //args_.input = "../../asset/sponza/lod.obj";
        //args_.input = "../../asset/suzanne/suzanne.obj";
        args_.input = "../../asset/bunny/bunny.obj";

#if 0
        if (!ParseArguments(argc, argv, cmd)) {
            return 0;
        }
#endif

        LoadObj();

        ctxt_.InitAllTextureAsGLTexture();

        // TODO
        aten::vec3 pos(0, 1, 3);
        aten::vec3 at(0, 1, 0);
        float vfov = float(45);

        camera_.init(
            pos,
            at,
            aten::vec3(0, 1, 0),
            vfov,
            WIDTH, HEIGHT);

        viewer_.init(
            WIDTH, HEIGHT,
            "voxelviewer_vs.glsl",
            "voxelviewer_fs.glsl");

        rasterizer_.init(
            WIDTH, HEIGHT,
            "../shader/drawobj_vs.glsl",
            "../shader/drawobj_fs.glsl");

        return true;
    }

    bool Run()
    {
        if (is_camera_dirty_) {
            camera_.update();

            auto camparam = camera_.param();
            camparam.znear = float(0.1);
            camparam.zfar = float(10000.0);

            is_camera_dirty_ = false;
        }

        // TODO
        auto sbvh = reinterpret_cast<aten::sbvh*>(objs_[0]->getInternalAccelerator());

        if (voxels_.empty()) {
            auto maxDepth = sbvh->getMaxDepth();
            voxels_.resize(maxDepth);

            // NOTE
            // nodes[0] is top layer.
            const auto& nodes = scene_.getAccel()->getNodes();
            viewer_.bringVoxels(nodes[1], voxels_);
        }

        aten::RasterizeRenderer::clearBuffer(
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Stencil,
            aten::vec4(0, 0.5f, 1.0f, 1.0f),
            1.0f,
            0);

        viewer_.draw(
            ctxt_,
            &camera_,
            voxels_,
            is_wireframe_,
            draw_voxel_depth_);

        if (is_draw_wesh_) {
            for (auto& obj : objs_) {
                rasterizer_.drawObject(ctxt_, *obj, &camera_, false);
            }
        }

        {
            ImGui::SliderInt("Depth", &draw_voxel_depth_, 1, sbvh->getMaxDepth());
            ImGui::Checkbox("Wireframe,", &is_wireframe_);
            ImGui::Checkbox("Draw mesh,", &is_draw_wesh_);
        }

        return true;
    }

    void OnClose()
    {
    }

    void OnMouseBtn(bool left, bool press, int32_t x, int32_t y)
    {
        is_mouse_l_btn_down_ = false;
        is_mouse_r_btn_down_ = false;

        if (press)
        {
            prev_mouse_pos_x_ = x;
            prev_mouse_pos_y_ = y;

            is_mouse_l_btn_down_ = left;
            is_mouse_r_btn_down_ = !left;
        }
    }

    void OnMouseMove(int32_t x, int32_t y)
    {
        if (is_mouse_l_btn_down_)
        {
            aten::CameraOperator::Rotate(
                camera_,
                WIDTH, HEIGHT,
                prev_mouse_pos_x_, prev_mouse_pos_y_,
                x, y);
            is_camera_dirty_ = true;
        }
        else if (is_mouse_r_btn_down_)
        {
            aten::CameraOperator::move(
                camera_,
                prev_mouse_pos_x_, prev_mouse_pos_y_,
                x, y,
                float(0.001));
            is_camera_dirty_ = true;
        }

        prev_mouse_pos_x_ = x;
        prev_mouse_pos_y_ = y;
    }

    void OnMouseWheel(int32_t delta)
    {
        aten::CameraOperator::Dolly(camera_, delta * float(0.1));
        is_camera_dirty_ = true;
    }

    void OnKey(bool press, aten::Key key)
    {
        static const float offset_base = float(0.1);

        if (press)
        {
            if (key == aten::Key::Key_F1)
            {
                will_show_gui_ = !will_show_gui_;
                return;
            }
        }

        auto offset = offset_base;

        if (press)
        {
            switch (key)
            {
            case aten::Key::Key_W:
            case aten::Key::Key_UP:
                aten::CameraOperator::MoveForward(camera_, offset);
                break;
            case aten::Key::Key_S:
            case aten::Key::Key_DOWN:
                aten::CameraOperator::MoveForward(camera_, -offset);
                break;
            case aten::Key::Key_D:
            case aten::Key::Key_RIGHT:
                aten::CameraOperator::MoveRight(camera_, offset);
                break;
            case aten::Key::Key_A:
            case aten::Key::Key_LEFT:
                aten::CameraOperator::MoveRight(camera_, -offset);
                break;
            case aten::Key::Key_Z:
                aten::CameraOperator::MoveUp(camera_, offset);
                break;
            case aten::Key::Key_X:
                aten::CameraOperator::MoveUp(camera_, -offset);
                break;
            default:
                break;
            }

            is_camera_dirty_ = true;
        }
    }

    aten::context& GetContext()
    {
        return ctxt_;
    }

private:
    bool ParseArguments(int32_t argc, char* argv[])
    {
        cmdline::parser cmd;
        {
            cmd.add<std::string>("input", 'i', "input filename", true);
            cmd.add<std::string>("output", 'o', "output filename base", false, "result");

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
            // TODO
            args_.output = "result.sbvh";
        }

        return true;
    }

    // TODO
    void LoadObj()
    {
        aten::MaterialParameter mtrlParam;
        mtrlParam.type = aten::MaterialType::GGX;
        mtrlParam.baseColor = aten::vec3(0.7, 0.7, 0.7);
        mtrlParam.standard.ior = 0.2;
        mtrlParam.standard.roughness = 0.2;

        auto mtrl = ctxt_.CreateMaterialWithMaterialParameter(
            "m",
            mtrlParam,
            nullptr, nullptr, nullptr);

        objs_ = aten::ObjLoader::Load(args_.input, ctxt_);

        for (auto& obj : objs_) {
            auto instance = aten::TransformableFactory::createInstance<aten::PolygonObject>(ctxt_, obj, aten::mat4::Identity);
            scene_.add(instance);
        }

        scene_.build(ctxt_);
    }

    struct Args {
        std::string input;
        std::string output;
    } args_;

    aten::context ctxt_;

    std::vector<std::shared_ptr<aten::PolygonObject>> objs_;
    aten::AcceleratedScene<aten::sbvh> scene_;

    aten::RasterizeRenderer rasterizer_;

    std::vector<std::vector<aten::ThreadedSbvhNode>> voxels_;
    VoxelViewer viewer_;

    aten::PinholeCamera camera_;
    bool is_camera_dirty_{ false };

    bool will_show_gui_{ true };

    int32_t draw_voxel_depth_{ 1 };
    bool is_draw_wesh_{ false };
    bool is_wireframe_{ false };

    bool is_mouse_l_btn_down_{ false };
    bool is_mouse_r_btn_down_{ false };
    int32_t prev_mouse_pos_x_{ 0 };
    int32_t prev_mouse_pos_y_{ 0 };
};

int32_t main(int32_t argc, char* argv[])
{
    aten::SetCurrentDirectoryFromExe();

    auto app = std::make_shared<VoxelViewerApp>();

    auto wnd = std::make_shared<aten::window>();

    aten::window::MesageHandlers handlers;
    handlers.OnRun = [&app]() { return app->Run(); };
    handlers.OnClose = [&app]() { app->OnClose(); };
    handlers.OnMouseBtn = [&app](bool left, bool press, int32_t x, int32_t y) { app->OnMouseBtn(left, press, x, y); };
    handlers.OnMouseMove = [&app](int32_t x, int32_t y) { app->OnMouseMove(x, y);  };
    handlers.OnMouseWheel = [&app](int32_t delta) { app->OnMouseWheel(delta); };
    handlers.OnKey = [&app](bool press, aten::Key key) { app->OnKey(press, key); };

    auto id = wnd->Create(
        WIDTH, HEIGHT, TITLE,
        handlers);

    if (id >= 0) {
        app->GetContext().SetIsWindowInitialized(true);
    }
    else {
        AT_ASSERT(false);
        return 1;
    }

    app->Init(argc, argv);

    wnd->Run();

    app.reset();

    wnd->Terminate();

    return 1;
}
