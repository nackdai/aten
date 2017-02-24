#include <vector>
#include "aten.h"
#include "scenedefs.h"

static int WIDTH = 640;
static int HEIGHT = 480;
static const char* TITLE = "app";

//#define ENABLE_DOF

#ifdef ENABLE_DOF
static aten::ThinLensCamera g_camera;
#else
static aten::PinholeCamera g_camera;
#endif

//static aten::AcceledScene<aten::LinearList> g_scene;
static aten::AcceledScene<aten::bvhnode> g_scene;

//static aten::StaticColorBG g_bg(aten::vec3(0, 0, 0));
static aten::envmap g_bg;
static aten::texture* g_envmap;

//static aten::RayTracing g_tracer;
//static aten::PathTracing g_tracer;
static aten::PathTracingOpt g_tracer;

static std::vector<aten::vec3> g_buffer;
static std::vector<aten::color> g_dst;

static bool isExportedHdr = false;

#ifdef ENABLE_OMP
static uint32_t g_threadnum = 8;
#else
static uint32_t g_threadnum = 1;
#endif

void display()
{
	aten::Destination dst;
	{
		dst.width = WIDTH;
		dst.height = HEIGHT;
		dst.maxDepth = 5;
		dst.russianRouletteDepth = 3;
		dst.sample = 100;
		dst.buffer = &g_buffer[0];
	}

	aten::timer timer;
	timer.begin();

	// Trace rays.
	g_tracer.render(dst, &g_scene, &g_camera);

	auto elapsed = timer.end();
	AT_PRINTF("Elapsed %f[ms]\n", elapsed);

	if (!isExportedHdr) {
		isExportedHdr = true;

		// Export to hdr format.
		aten::HDRExporter::save(
			"result.hdr",
			&g_buffer[0],
			WIDTH, HEIGHT);
	}

#if 0
	// Do tonemap.
	aten::Tonemap::doTonemap(
		WIDTH, HEIGHT,
		&g_buffer[0],
		&g_dst[0]);

	aten::visualizer::render(&g_dst[0]);
#else
	aten::visualizer::render(&g_buffer[0], g_camera.needRevert());
#endif
}

int main(int argc, char* argv[])
{
	aten::random::init();

#if 0
	aten::MicrofacetBlinn blinn(aten:: vec3(1, 1, 1), 1, 1.5);
	aten::vec3 normal(0, 1, 0);
	aten::vec3 in(1, -1, 0);
	in.normalize();

	aten::XorShift rnd(0);
	aten::UniformDistributionSampler sampler(&rnd);

	auto ddd = Rad2Deg(acos(dot(normal, -in)));
	AT_PRINTF("in : %f\n", ddd);

	for (int i = 0; i < 100; i++) {
		auto wo = blinn.sampleDirection(in, normal, &sampler);
		auto xxx = Rad2Deg(acos(dot(normal, wo)));
		AT_PRINTF("out : %f\n", xxx);
	}
#endif

	aten::timer::init();
	aten::thread::setThreadNum(g_threadnum);

	aten::window::init(WIDTH, HEIGHT, TITLE);

	aten::visualizer::init(WIDTH, HEIGHT, aten::PixelFormat::rgba32f);
	//aten::visualizer::init(WIDTH, HEIGHT, aten::PixelFormat::rgba8);

	aten::SimpleRender shader;
	shader.init(
		WIDTH, HEIGHT,
		"../shader/vs.glsl",
		"../shader/fs.glsl");

	aten::TonemapRender tonemap;
	tonemap.init(
		WIDTH, HEIGHT,
		"../shader/vs.glsl",
		"../shader/tonemapfs.glsl");

	aten::visualizer::setShader(&tonemap);

	aten::vec3 lookfrom;
	aten::vec3 lookat;

	Scene::getCameraPosAndAt(lookfrom, lookat);

#ifdef ENABLE_DOF
	g_camera.init(
		WIDTH, HEIGHT,
		lookfrom, lookat,
		aten::vec3(0, 1, 0),
		30.0,	// image sensor size
		40.0,	// distance image sensor to lens
		130.0,	// distance lens to object plane
		5.0,	// lens radius
		28.0);	// W scale
#else
	g_camera.init(
		lookfrom,
		lookat,
		aten::vec3(0, 1, 0),
		30,
		WIDTH, HEIGHT);
#endif

	// TODO
	aten::MaterialManager::addMaterial(
		"m1",
		new aten::MicrofacetBlinn(aten::vec3(0.7, 0.6, 0.5), 200, 0.2));
		//new aten::lambert(aten::vec3(0.7, 0.6, 0.5)));

	Scene::makeScene(&g_scene);

	g_scene.build();

	g_envmap = aten::ImageLoader::load("../../asset/studio015.hdr");
	//g_envmap = aten::ImageLoader::load("../../asset/harbor.hdr");
	g_bg.init(g_envmap);

	g_tracer.setBG(&g_bg);

	g_buffer.resize(WIDTH * HEIGHT);
	g_dst.resize(WIDTH * HEIGHT);

	aten::window::run(display);

	aten::window::terminate();
}