#if 1
#include <vector>
#include "aten.h"
#include "atenscene.h"
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
static aten::AcceleratedScene<aten::bvh> g_scene;

static aten::StaticColorBG g_staticbg(aten::vec3(0.25, 0.25, 0.25));
static aten::envmap g_bg;
static aten::texture* g_envmap;

static aten::RayTracing g_tracer;
//static aten::PathTracing g_tracer;
//static aten::BDPT g_tracer;
//static aten::BDPT2 g_tracer;
//static aten::SortedPathTracing g_tracer;
//static aten::ERPT g_tracer;
//static aten::PSSMLT g_tracer;
//static aten::GeometryInfoRenderer g_tracer;

//static aten::FilmProgressive g_buffer(WIDTH, HEIGHT);
static aten::Film g_buffer(WIDTH, HEIGHT);

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
		dst.maxDepth = 6;
		dst.russianRouletteDepth = 3;
		dst.startDepth = 0;
		dst.sample = 5;
		dst.mutation = 10;
		dst.mltNum = 10;
		dst.buffer = &g_buffer;
	}

	dst.geominfo.albedo_vis = &g_buffer;
	dst.geominfo.depthMax = 1000;

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
			g_buffer.image(),
			WIDTH, HEIGHT);
	}

	aten::visualizer::render(g_buffer.image(), g_camera.needRevert());
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

	Scene::makeScene(&g_scene);

	g_scene.build();

	//g_tracer.setVirtualLight(g_camera.getPos(), g_camera.getDir(), aten::vec3(36.0, 36.0, 36.0)* 2);

	g_envmap = aten::ImageLoader::load("../../asset/studio015.hdr");
	//g_envmap = aten::ImageLoader::load("../../asset/harbor.hdr");
	g_bg.init(g_envmap);

	aten::ImageBasedLight ibl(&g_bg);
	//g_scene.addImageBasedLight(&ibl);

	//g_tracer.setBG(&g_staticbg);

	//aten::NonLocalMeanFilter filter;
	//aten::BilateralFilter filter;
	//aten::visualizer::addPreProc(&filter);

	aten::window::run(display);

	aten::window::terminate();
}
#endif
