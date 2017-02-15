#include <vector>
#include "aten.h"

static int WIDTH = 640;
static int HEIGHT = 480;
static const char* TITLE = "app";

static aten::PinholeCamera g_camera;
static aten::AcceledScene<aten::LinearList> g_scene;

//static aten::StaticColorBG m_bg(aten::vec3(1, 1, 0));
static aten::envmap m_bg;
static aten::texture* g_envmap;

//static aten::RayTracing g_tracer;
static aten::PathTracing g_tracer;

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
	aten::visualizer::render(&g_buffer[0]);
#endif

	
}

void makeScene()
{
	auto light = new aten::sphere(
		aten::vec3(50.0, 90.0, 81.6),
		15.0,
		new aten::emissive(aten::vec3(36.0, 36.0, 36.0)));

	double r = 1e3;

	auto left = new aten::sphere(
		aten::vec3(r + 1, 40.8, 81.6),
		r,
		new aten::diffuse(aten::vec3(0.75f, 0.25f, 0.25f)));

	auto right = new aten::sphere(
		aten::vec3(-r + 99, 40.8, 81.6),
		r,
		new aten::diffuse(aten::vec3(0.25, 0.25, 0.75)));

	auto wall = new aten::sphere(
		aten::vec3(50, 40.8, r),
		r,
		new aten::diffuse(aten::vec3(0.75, 0.75, 0.75)));

	auto floor = new aten::sphere(
		aten::vec3(50, r, 81.6),
		r,
		new aten::diffuse(aten::vec3(0.75, 0.75, 0.75)));

	auto ceil = new aten::sphere(
		aten::vec3(50, -r + 81.6, 81.6),
		r,
		new aten::diffuse(aten::vec3(0.75, 0.75, 0.75)));

	// —Î‹….
	auto green = new aten::sphere(
		aten::vec3(65, 20, 20),
		20,
		new aten::diffuse(aten::vec3(0.25, 0.75, 0.25)));

	// ‹¾.
	auto mirror = new aten::sphere(
		aten::vec3(27, 16.5, 47), 
		16.5, 
		new aten::specular(aten::vec3(0.99, 0.99, 0.99)));

	// ƒKƒ‰ƒX.
	auto glass = new aten::sphere(
		aten::vec3(77, 16.5, 78),
		16.5,
		new aten::refraction(aten::vec3(0.99, 0.99, 0.99), 1.5));

#if 1
	g_scene.add(light);
	g_scene.add(left);
	g_scene.add(right);
	g_scene.add(wall);
	g_scene.add(floor);
	g_scene.add(ceil);
	g_scene.add(green);
	g_scene.add(mirror);
	g_scene.add(glass);

	g_scene.addLight(light);
#endif
}

int main(int argc, char* argv[])
{
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

	aten::vec3 lookfrom(50.0, 52.0, 295.6);
	aten::vec3 lookat(50.0, 40.8, 119.0);

	g_camera.init(
		lookfrom,
		lookat,
		aten::vec3(0, 1, 0),
		30,
		WIDTH, HEIGHT);

	makeScene();

	g_scene.build();

	g_envmap = aten::ImageLoader::load("studio015.hdr");
	m_bg.init(g_envmap);

	g_tracer.setBG(&m_bg);

	g_buffer.resize(WIDTH * HEIGHT);
	g_dst.resize(WIDTH * HEIGHT);

	aten::window::run(display);

	aten::window::terminate();
}