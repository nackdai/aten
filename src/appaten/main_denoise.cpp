#if 0
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
static aten::AcceledScene<aten::bvh> g_scene;

static aten::StaticColorBG g_staticbg(aten::vec3(0.25, 0.25, 0.25));
static aten::envmap g_bg;
static aten::texture* g_envmap;

static aten::PathTracing g_pathtracer;
static aten::RayTracing g_raytracer;
static aten::GeometryInfoRenderer g_geotracer;

static std::vector<aten::vec4> g_directBuffer(WIDTH * HEIGHT);
static std::vector<aten::vec4> g_indirectBuffer(WIDTH * HEIGHT);
static std::vector<aten::vec4> g_varIndirectBuffer(WIDTH * HEIGHT);
static std::vector<aten::vec4> g_nml_depth_Buffer(WIDTH * HEIGHT);

#ifdef ENABLE_OMP
static uint32_t g_threadnum = 8;
#else
static uint32_t g_threadnum = 1;
#endif

#define VFI
static aten::PointLight virtualLight;

#ifdef VFI
aten::VirtualFlashImage denoiser;
#else
aten::PracticalNoiseReduction denoiser;
#endif

void display()
{
#ifdef VFI
	aten::vec4* image = &g_directBuffer[0];
	aten::vec4* varImage = &g_indirectBuffer[0];
	aten::vec4* flash = &g_varIndirectBuffer[0];
	aten::vec4* varFlash = &g_nml_depth_Buffer[0];

	{
		aten::Destination dst;
		{
			dst.width = WIDTH;
			dst.height = HEIGHT;
			dst.maxDepth = 6;
			dst.russianRouletteDepth = 3;
			dst.startDepth = 0;
			dst.sample = 16;
			dst.mutation = 10;
			dst.mltNum = 10;
			dst.buffer = image;
			dst.variance = varImage;
		}

		g_pathtracer.setVirtualLight(nullptr, aten::vec3(0));
		g_pathtracer.render(dst, &g_scene, &g_camera);
	}

	{
		aten::Destination dst;
		{
			dst.width = WIDTH;
			dst.height = HEIGHT;
			dst.maxDepth = 6;
			dst.russianRouletteDepth = 3;
			dst.startDepth = 0;
			dst.sample = 16;
			dst.mutation = 10;
			dst.mltNum = 10;
			dst.buffer = flash;
			dst.variance = varFlash;
		}

		virtualLight.setPos(g_camera.getPos());
		virtualLight.setLe(aten::vec3(36.0, 36.0, 36.0) * 2);
		g_pathtracer.setVirtualLight(&virtualLight, g_camera.getDir());
		g_pathtracer.render(dst, &g_scene, &g_camera);
	}

	denoiser.setParam(
		16,
		varImage, flash, varFlash);

	aten::visualizer::render(image, g_camera.needRevert());
#else
	{
		aten::Destination dst;
		{
			dst.width = WIDTH;
			dst.height = HEIGHT;
			dst.maxDepth = 6;
			dst.russianRouletteDepth = 3;
			dst.startDepth = 1;
			dst.sample = 40;
			dst.mutation = 10;
			dst.mltNum = 10;
			dst.buffer = &g_indirectBuffer[0];
			dst.variance = &g_varIndirectBuffer[0];
		}

		g_pathtracer.render(dst, &g_scene, &g_camera);
	}

	{
		aten::Destination dst;
		{
			dst.width = WIDTH;
			dst.height = HEIGHT;
			dst.maxDepth = 6;
			dst.buffer = &g_directBuffer[0];
		}

		g_raytracer.render(dst, &g_scene, &g_camera);
		//g_pathtracer.render(dst, &g_scene, &g_camera);
	}

	
	{
		aten::Destination dst;
		{
			dst.width = WIDTH;
			dst.height = HEIGHT;
			dst.geominfo.nml_depth = &g_nml_depth_Buffer[0];
			dst.geominfo.depthMax = 1000;
			dst.geominfo.needNormalize = false;
		}

		g_geotracer.render(dst, &g_scene, &g_camera);
	}

	denoiser.setBuffers(
		&g_directBuffer[0],
		&g_indirectBuffer[0],
		&g_varIndirectBuffer[0],
		&g_nml_depth_Buffer[0]);

	aten::visualizer::render(&g_indirectBuffer[0], g_camera.needRevert());
#endif
}

int main(int argc, char* argv[])
{
	aten::random::init();

	aten::timer::init();
	aten::thread::setThreadNum(g_threadnum);

	aten::window::init(WIDTH, HEIGHT, TITLE);

	aten::visualizer::init(WIDTH, HEIGHT);

	//aten::BilateralFilter filter;
	//aten::NonLocalMeanFilter filter;

	aten::visualizer::addPreProc(&denoiser);
	//aten::visualizer::addPreProc(&filter);

	aten::TonemapPostProc tonemap;
	tonemap.init(
		WIDTH, HEIGHT,
		"../shader/vs.glsl",
		"../shader/tonemap_fs.glsl");

	aten::GammaCorrection gamma;
	gamma.init(
		WIDTH, HEIGHT,
		"../shader/vs.glsl",
		"../shader/gamma_fs.glsl");

	aten::Blitter blitter;
	blitter.init(
		WIDTH, HEIGHT,
		"../shader/vs.glsl",
		"../shader/fs.glsl");
	blitter.setIsRenderRGB(true);

	aten::visualizer::addPostProc(&gamma);
	//aten::visualizer::addPostProc(&tonemap);
	//aten::visualizer::addPostProc(&blitter);

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

	aten::window::run(display);

	aten::window::terminate();
}
#endif
