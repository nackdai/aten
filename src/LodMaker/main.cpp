#include "aten.h"
#include "atenscene.h"

static const int WIDTH = 1280;
static const int HEIGHT = 720;

static const char* TITLE = "LodMaker";

static aten::PinholeCamera g_camera;
static bool g_isCameraDirty = false;

static bool g_willShowGUI = true;
static bool g_willTakeScreenShot = false;
static int g_cntScreenShot = 0;

static bool g_isMouseLBtnDown = false;
static bool g_isMouseRBtnDown = false;
static int g_prevX = 0;
static int g_prevY = 0;

void onRun()
{

}

void onClose()
{

}

void onMouseBtn(bool left, bool press, int x, int y)
{
	g_isMouseLBtnDown = false;
	g_isMouseRBtnDown = false;

	if (press) {
		g_prevX = x;
		g_prevY = y;

		g_isMouseLBtnDown = left;
		g_isMouseRBtnDown = !left;
	}
}

void onMouseMove(int x, int y)
{
	if (g_isMouseLBtnDown) {
		aten::CameraOperator::rotate(
			g_camera,
			WIDTH, HEIGHT,
			g_prevX, g_prevY,
			x, y);
		g_isCameraDirty = true;
	}
	else if (g_isMouseRBtnDown) {
		aten::CameraOperator::move(
			g_camera,
			g_prevX, g_prevY,
			x, y,
			real(0.001));
		g_isCameraDirty = true;
	}

	g_prevX = x;
	g_prevY = y;
}

void onMouseWheel(int delta)
{
	aten::CameraOperator::dolly(g_camera, delta * real(0.1));
	g_isCameraDirty = true;
}

void onKey(bool press, aten::Key key)
{
	static const real offset = real(0.1);

	if (press) {
		if (key == aten::Key::Key_F1) {
			g_willShowGUI = !g_willShowGUI;
			return;
		}
		else if (key == aten::Key::Key_F2) {
			g_willTakeScreenShot = true;
			return;
		}
	}

	if (press) {
		switch (key) {
		case aten::Key::Key_W:
		case aten::Key::Key_UP:
			aten::CameraOperator::moveForward(g_camera, offset);
			break;
		case aten::Key::Key_S:
		case aten::Key::Key_DOWN:
			aten::CameraOperator::moveForward(g_camera, -offset);
			break;
		case aten::Key::Key_D:
		case aten::Key::Key_RIGHT:
			aten::CameraOperator::moveRight(g_camera, offset);
			break;
		case aten::Key::Key_A:
		case aten::Key::Key_LEFT:
			aten::CameraOperator::moveRight(g_camera, -offset);
			break;
		case aten::Key::Key_Z:
			aten::CameraOperator::moveUp(g_camera, offset);
			break;
		case aten::Key::Key_X:
			aten::CameraOperator::moveUp(g_camera, -offset);
			break;
		default:
			break;
		}

		g_isCameraDirty = true;
	}
}

int main()
{
	aten::window::init(
		WIDTH, HEIGHT,
		TITLE,
		onClose,
		onMouseBtn,
		onMouseMove,
		onMouseWheel,
		onKey);

	// TODO
	aten::vec3 pos(0, 0, 10);
	aten::vec3 at(0, 0, 1);
	real vfov = real(45);

	g_camera.init(
		pos,
		at,
		aten::vec3(0, 1, 0),
		vfov,
		WIDTH, HEIGHT);

	aten::window::run(onRun);

	aten::window::terminate();

	return 1;
}
