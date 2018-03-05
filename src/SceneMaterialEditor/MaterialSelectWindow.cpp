#include "MaterialSelectWindow.h"

#include "atenscene.h"

aten::RasterizeRenderer MaterialSelectWindow::s_rasterizer;

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

void MaterialSelectWindow::onRun(aten::window* window)
{
	if (s_isCameraDirty) {
		s_camera.update();

		auto camparam = s_camera.param();
		camparam.znear = real(0.1);
		camparam.zfar = real(10000.0);

		s_isCameraDirty = false;
	}

	s_rasterizer.draw(
		s_obj,
		&s_camera,
		false,
		&s_fbo);

	s_fbo.bindAsTexture();
	s_visualizer->render(s_fbo.getTexHandle(), false);
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

void MaterialSelectWindow::onKey(bool press, aten::Key key)
{
	static const real offset = real(0.1);

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
aten::object* loadObj(const char* input)
{
	std::vector<aten::object*> objs;

	aten::ObjLoader::load(objs, input);

	// NOTE
	// ‚P‚Â‚µ‚©‚ä‚é‚³‚È‚¢.
	AT_ASSERT(objs.size() == 1);

	auto obj = objs[0];

	return obj;
}

bool MaterialSelectWindow::init(
	int width, int height,
	const char* title,
	const char* input)
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

	s_obj = loadObj(input);

	s_obj->buildForRasterizeRendering();

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
		"../shader/drawobj_vs.glsl",
		"../shader/drawobj_fs.glsl");

	s_visualizer = aten::visualizer::init(s_width, s_height);

	s_blitter.init(
		s_width, s_height,
		"../shader/fullscreen_vs.glsl",
		"../shader/fullscreen_fs.glsl");

	s_visualizer->addPostProc(&s_blitter);

	s_fbo.asMulti(2);
	s_fbo.init(s_width, s_height, aten::PixelFormat::rgba8, true);

	return true;
}
