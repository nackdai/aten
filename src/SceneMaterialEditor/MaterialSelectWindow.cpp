#include "MaterialSelectWindow.h"

#include "atenscene.h"

aten::RasterizeRenderer MaterialSelectWindow::s_rasterizer;

static aten::context s_ctxt;

aten::object* MaterialSelectWindow::s_obj = nullptr;

aten::Blitter MaterialSelectWindow::s_blitter;
aten::visualizer* MaterialSelectWindow::s_visualizer = nullptr;

aten::FBO MaterialSelectWindow::s_fbo;

aten::PinholeCamera MaterialSelectWindow::s_camera;
bool MaterialSelectWindow::s_isCameraDirty = false;

bool MaterialSelectWindow::s_willTakeScreenShot = false;
int MaterialSelectWindow::s_cntScreenShot = 0;

bool MaterialSelectWindow::s_isMouseLBtnDown = false;
bool MaterialSelectWindow::s_isMouseRBtnDown = false;
int MaterialSelectWindow::s_prevX = 0;
int MaterialSelectWindow::s_prevY = 0;

int MaterialSelectWindow::s_width = 0;
int MaterialSelectWindow::s_height = 0;

bool MaterialSelectWindow::s_willPick = false;
bool MaterialSelectWindow::s_pick = false;

std::vector<aten::TColor<uint8_t, 4>> MaterialSelectWindow::s_attrib;

int MaterialSelectWindow::s_pickedMtrlId = 0;

MaterialSelectWindow::FuncPickMtrlIdNotifier MaterialSelectWindow::s_pickMtrlIdNotifier = nullptr;

template <typename T>
class blink {
public:
    blink(T _min, T _max, T step)
        : m_step(step)
    {
        m_min = std::min(_min, _max);
        m_max = std::max(_min, _max);
        m_value = m_min;
    }
    ~blink() {}

public:
    float update()
    {
        m_value += m_isForward ? m_step : -m_step;

        if (m_value >= m_max) {
            m_isForward = false;
        }
        else if (m_value <= m_min) {
            m_isForward = true;
        }

        m_value = aten::clamp(m_value, m_min, m_max);

        float normalized = (m_value - m_min) / (float)(m_max - m_min);

        return normalized;
    }

private:
    T m_value;
    T m_min{ (T)0 };
    T m_max{ (T)0 };
    T m_step{ (T)0 };

    bool m_isForward{ true };
};

void MaterialSelectWindow::notifyChangeMtrlId(int mtrlid)
{
    s_pickedMtrlId = mtrlid;
}

void MaterialSelectWindow::onRun(aten::window* window)
{
    if (s_isCameraDirty) {
        s_camera.update();

        auto camparam = s_camera.param();
        camparam.znear = real(0.1);
        camparam.zfar = real(10000.0);

        s_isCameraDirty = false;
    }

    static blink<float> s_blinker(0.0f, 1.0f, 0.05f);
    auto t = s_blinker.update();

    aten::mat4 mtxL2W;
    //mtxL2W.asRotateByX(Deg2Rad(-90));

    s_rasterizer.drawObject(
        s_ctxt,
        *s_obj,
        &s_camera,
        false,
        mtxL2W,
        &s_fbo,
        [&](aten::shader& shd, const aten::vec3& color, const aten::texture* albedo, int mtrlid)
    {
        if (s_pickedMtrlId == mtrlid) {
            shd.setUniformBool("isSelected", true);
            shd.setUniformFloat("time", t);
        }
        else {
            shd.setUniformBool("isSelected", false);
        }
    });

    s_fbo.bindAsTexture();
    s_visualizer->render(s_fbo.getTexHandle(), false);

    if (s_pick) {
        aten::visualizer::getTextureData(s_fbo.getTexHandle(1), s_attrib);

        // NOTE
        // up-side-down.
        int pos = (s_height - 1 - s_prevY) * s_width + s_prevX;
        auto attrib = s_attrib[pos];

        int mtrlid = (int)attrib.r();
        mtrlid -= 1;

#if 0
        aten::material* mtrl = nullptr;
        if (mtrlid >= 0) {
            mtrl = aten::material::getMaterial(mtrlid);
        }

        AT_PRINTF(
            "(%d, %d)[%d]->(%s)\n", 
            s_prevX, s_prevY, 
            mtrlid,
            mtrl ? mtrl->name() : "none");
#endif

        if (mtrlid >= 0) {
            s_pickMtrlIdNotifier(mtrlid);
            s_pickedMtrlId = mtrlid;
        }

        s_pick = false;
    }
}

void MaterialSelectWindow::onClose()
{

}

void MaterialSelectWindow::onMouseBtn(bool left, bool press, int x, int y)
{
    s_isMouseLBtnDown = false;
    s_isMouseRBtnDown = false;

    if (press) {
        s_prevX = x;
        s_prevY = y;

        s_isMouseLBtnDown = left;
        s_isMouseRBtnDown = !left;

        if (s_isMouseLBtnDown) {
            if (s_willPick) {
                s_pick = true;
            }
            else {
                s_pick = false;
            }
            s_willPick = false;
        }
    }
}

void MaterialSelectWindow::onMouseMove(int x, int y)
{
    if (s_isMouseLBtnDown) {
        aten::CameraOperator::rotate(
            s_camera,
            s_width, s_height,
            s_prevX, s_prevY,
            x, y);
        s_isCameraDirty = true;
    }
    else if (s_isMouseRBtnDown) {
        aten::CameraOperator::move(
            s_camera,
            s_prevX, s_prevY,
            x, y,
            real(0.001));
        s_isCameraDirty = true;
    }

    s_prevX = x;
    s_prevY = y;
}

void MaterialSelectWindow::onMouseWheel(int delta)
{
    aten::CameraOperator::dolly(s_camera, delta * real(0.1));
    s_isCameraDirty = true;
}

real offset = real(0.1);

void MaterialSelectWindow::onKey(bool press, aten::Key key)
{
    if (press) {
        if (key == aten::Key::Key_CONTROL) {
            s_willPick = true;
            return;
        }
        else if (key == aten::Key::Key_SHIFT) {
            offset *= 10.0f;
            return;
        }
    }
    else {
        if (key == aten::Key::Key_CONTROL) {
            s_willPick = false;
            s_pick = false;
            return;
        }
    }

    if (press) {
        switch (key) {
        case aten::Key::Key_W:
        case aten::Key::Key_UP:
            aten::CameraOperator::moveForward(s_camera, offset);
            break;
        case aten::Key::Key_S:
        case aten::Key::Key_DOWN:
            aten::CameraOperator::moveForward(s_camera, -offset);
            break;
        case aten::Key::Key_D:
        case aten::Key::Key_RIGHT:
            aten::CameraOperator::moveRight(s_camera, offset);
            break;
        case aten::Key::Key_A:
        case aten::Key::Key_LEFT:
            aten::CameraOperator::moveRight(s_camera, -offset);
            break;
        case aten::Key::Key_Z:
            aten::CameraOperator::moveUp(s_camera, offset);
            break;
        case aten::Key::Key_X:
            aten::CameraOperator::moveUp(s_camera, -offset);
            break;
        default:
            break;
        }

        s_isCameraDirty = true;
    }
}

// TODO
aten::object* loadObj(
    const char* objpath,
    const char* mtrlpath)
{
    std::string pathname;
    std::string extname;
    std::string filename;

    aten::getStringsFromPath(
        objpath,
        pathname,
        extname,
        filename);

    aten::ImageLoader::setBasePath(pathname);

    if (mtrlpath) {
        aten::MaterialLoader::load(mtrlpath, s_ctxt);
    }

    std::vector<aten::object*> objs;

    aten::ObjLoader::load(objs, objpath, s_ctxt);

    // NOTE
    // ‚P‚Â‚µ‚©‚ä‚é‚³‚È‚¢.
    AT_ASSERT(objs.size() == 1);

    auto obj = objs[0];

    return obj;
}

bool MaterialSelectWindow::init(
    int width, int height,
    const char* title,
    const char* objpath,
    const char* mtrlpath/*= nullptr*/)
{
    s_width = width;
    s_height = height;

    auto wnd = aten::window::init(
        s_width, s_height,
        title,
        MaterialSelectWindow::onRun,
        MaterialSelectWindow::onClose,
        MaterialSelectWindow::onMouseBtn,
        MaterialSelectWindow::onMouseMove,
        MaterialSelectWindow::onMouseWheel,
        MaterialSelectWindow::onKey);

    wnd->asCurrent();

    s_obj = loadObj(objpath, mtrlpath);

    s_ctxt.initAllTexAsGLTexture();

    s_obj->buildForRasterizeRendering(s_ctxt);

    // TODO
    aten::vec3 pos(0, 1, 10);
    aten::vec3 at(0, 1, 1);
    real vfov = real(45);

    s_camera.init(
        pos,
        at,
        aten::vec3(0, 1, 0),
        vfov,
        s_width, s_height);

    s_rasterizer.init(
        s_width, s_height,
        "./drawobj_vs.glsl",
        "./drawobj_fs.glsl");

    s_visualizer = aten::visualizer::init(s_width, s_height);

    s_blitter.init(
        s_width, s_height,
        "../shader/fullscreen_vs.glsl",
        "../shader/fullscreen_fs.glsl");

    s_visualizer->addPostProc(&s_blitter);

    s_fbo.asMulti(2);
    s_fbo.init(s_width, s_height, aten::PixelFormat::rgba8, true);

    s_attrib.resize(s_width * s_height);

    return true;
}
