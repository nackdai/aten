#if 1
#include "aten.h"
#include "atenscene.h"

static const char* TITLE = "appaten";

//static aten::RayTracing g_tracer;
static aten::PathTracing g_tracer;
//static aten::SortedPathTracing g_tracer;
//static aten::ERPT g_tracer;
//static aten::PSSMLT g_tracer;
//static aten::GeometryInfoRenderer g_tracer;

static int WIDTH = 0;
static int HEIGHT = 0;

static aten::Film* g_film = nullptr;
static aten::SceneLoader::SceneInfo sceneinfo;

static bool isExportedHdr = false;

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
	g_tracer.render(sceneinfo.dst, sceneinfo.scene, sceneinfo.camera);

	auto elapsed = timer.end();
	AT_PRINTF("Elapsed %f[ms]\n", elapsed);

	if (!isExportedHdr) {
		isExportedHdr = true;

		// Export to hdr format.
		aten::HDRExporter::save(
			"result.hdr",
			g_film->image(),
			WIDTH, HEIGHT);
	}

	aten::visualizer::render(g_film->image(), sceneinfo.camera->needRevert());
}

int main(int argc, char* argv[])
{
	std::string scenefile;

	if (argc > 1) {
		scenefile = argv[1];
	}
	else {
		// TODO
		return 1;
	}

	aten::random::init();

	aten::timer::init();
	aten::thread::setThreadNum(g_threadnum);

	try {
		sceneinfo = aten::SceneLoader::load(scenefile);

		WIDTH = sceneinfo.dst.width;
		HEIGHT = sceneinfo.dst.height;

		aten::window::init(WIDTH, HEIGHT, TITLE);

		aten::visualizer::init(WIDTH, HEIGHT);

		aten::Blitter blitter;
		blitter.init(
			WIDTH, HEIGHT,
			"../shader/vs.glsl",
			"../shader/fs.glsl");
		blitter.setIsRenderRGB(true);

		aten::TonemapPostProc tonemap;
		tonemap.init(
			WIDTH, HEIGHT,
			"../shader/vs.glsl",
			"../shader/tonemap_fs.glsl");

		aten::NonLocalMeanFilterShader nlmshd;
		nlmshd.init(
			WIDTH, HEIGHT,
			"../shader/vs.glsl",
			"../shader/nlm_fs.glsl");

		aten::BilateralFilterShader bishd;
		bishd.init(
			WIDTH, HEIGHT,
			"../shader/vs.glsl",
			"../shader/bilateral_fs.glsl");

		aten::BloomEffect bloom;
		bloom.init(
			WIDTH, HEIGHT,
			aten::rgba32f, aten::rgba32f,
			"../shader/vs.glsl",
			"../shader/bloomeffect_fs_4x4.glsl",
			"../shader/bloomeffect_fs_2x2.glsl",
			"../shader/bloomeffect_fs_HBlur.glsl",
			"../shader/bloomeffect_fs_VBlur.glsl",
			"../shader/bloomeffect_fs_Gauss.glsl",
			"../shader/bloomeffect_fs_Final.glsl");
		bloom.setParam(0.2f, 0.4f);

		aten::GammaCorrection gamma;
		gamma.init(
			WIDTH, HEIGHT,
			"../shader/vs.glsl",
			"../shader/gamma_fs.glsl");

		//aten::visualizer::addPostProc(&bishd);
		//aten::visualizer::addPostProc(&blitter);
		aten::visualizer::addPostProc(&gamma);
		//aten::visualizer::addPostProc(&tonemap);
		//aten::visualizer::addPostProc(&bloom);

		sceneinfo.scene->build();

		g_film = new aten::Film(WIDTH, HEIGHT);

		aten::window::run(display);

		aten::window::terminate();
	}
	catch (std::exception* e) {

	}
}
#endif
