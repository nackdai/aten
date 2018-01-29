#if 0
#include <vector>
#include "aten.h"
#include "atenscene.h"

int main(int argc, char* argv[])
{
	aten::window::SetCurrentDirectoryFromExe();

	aten::ImageLoader::setBasePath("../../asset/");

#if 0
	//aten::MaterialLoader::load("material.json");
	aten::MaterialLoader::load("material.xml");

	auto mtrl0 = aten::AssetManager::getMtrl("test");
	AT_ASSERT(mtrl0);

	auto mtrl1 = aten::AssetManager::getMtrl("test2");
	AT_ASSERT(mtrl1);

	auto mtrl2 = aten::AssetManager::getMtrl("test3");
	AT_ASSERT(mtrl2);
#endif

	auto info = aten::SceneLoader::load("scene.xml");
}
#else
#include "aten.h"
#include "atenscene.h"

static const char* TITLE = "appaten";

static aten::Renderer* g_tracer = nullptr;

static int WIDTH = 0;
static int HEIGHT = 0;

static aten::Film* g_film = nullptr;
static aten::SceneLoader::SceneInfo sceneinfo;

#ifdef ENABLE_OMP
static uint32_t g_threadnum = 8;
#else
static uint32_t g_threadnum = 1;
#endif

void display()
{
	sceneinfo.dst.buffer = g_film;

	aten::timer timer;
	timer.begin();

	// Trace rays.
	g_tracer->render(sceneinfo.dst, sceneinfo.scene, sceneinfo.camera);

	auto elapsed = timer.end();
	AT_PRINTF("Elapsed %f[ms]\n", elapsed);

	aten::visualizer::render(g_film->image(), sceneinfo.camera->needRevert());
}

int main(int argc, char* argv[])
{
	aten::window::SetCurrentDirectoryFromExe();

	aten::ImageLoader::setBasePath("../../asset/");
	aten::ObjLoader::setBasePath("../../asset/");

	std::string scenefile;

	if (argc > 1) {
		scenefile = argv[1];
	}
	else {
		scenefile = "scene.xml";
	}

	aten::random::init();

	aten::timer::init();
	aten::OMPUtil::setThreadNum(g_threadnum);

	try {
		sceneinfo = aten::SceneLoader::load(scenefile);

		WIDTH = sceneinfo.dst.width;
		HEIGHT = sceneinfo.dst.height;

		aten::window::init(WIDTH, HEIGHT, TITLE);

		aten::visualizer::init(WIDTH, HEIGHT);

		// TODO
		aten::GammaCorrection gamma;
		gamma.init(
			WIDTH, HEIGHT,
			"../shader/fullscreen_vs.glsl",
			"../shader/gamma_fs.glsl");
		aten::visualizer::addPostProc(&gamma);

		sceneinfo.scene->build();

		g_film = new aten::Film(WIDTH, HEIGHT);

		// TODO
		g_tracer = new aten::PathTracing();

		aten::window::run(display);

		aten::window::terminate();
	}
	catch (std::exception* e) {

	}
}
#endif
