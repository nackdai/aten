#include "aten.h"

static int WIDTH = 640;
static int HEIGHT = 480;
static const char* TITLE = "app";

static aten::PinholeCamera g_camera;

void display()
{

}

int main(int argc, char* argv[])
{
	aten::window::init(WIDTH, HEIGHT, TITLE);

	aten::visualizer::init(
		WIDTH, HEIGHT,
		"../shader/vs.glsl",
		"../shaderfvs.glsl");

	//g_camera.init();

	aten::window::run(display);

	aten::window::terminate();
}