#include <vector>
#include "aten.h"

static int WIDTH = 640;
static int HEIGHT = 480;
static const char* TITLE = "app";

static aten::PinholeCamera g_camera;
static aten::AcceledScene<aten::LinearList> g_scene;
static aten::RayTracer g_tracer;

static std::vector<aten::vec3> g_buffer;
static std::vector<aten::color> g_dst;

static bool isExportedHdr = false;

void display()
{
	aten::visualizer::beginRender();

	aten::Destination dst;
	{
		dst.width = WIDTH;
		dst.height = HEIGHT;
		dst.buffer = &g_buffer[0];
	}

	// Trace rays.
	g_tracer.render(dst, &g_scene, &g_camera);

	if (!isExportedHdr) {
		isExportedHdr = true;

		// Export to hdr format.
		aten::HDRExporter::save(
			"result.hdr",
			&g_buffer[0],
			WIDTH, HEIGHT);
	}

	// Do tonemap.
	aten::Tonemap::doTonemap(
		WIDTH, HEIGHT,
		&g_buffer[0],
		&g_dst[0]);

	aten::visualizer::endRender(&g_dst[0]);
}

void makeScene()
{
	auto light = new aten::sphere(
		aten::vec3(50.0, 75.0, 81.6),
		5.0,
		new aten::emissive(aten::vec3(12.0, 12.0, 12.0)));

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

	g_scene.add(light);
	g_scene.add(left);
	g_scene.add(right);
	g_scene.add(wall);
	g_scene.add(floor);
	g_scene.add(ceil);

	g_scene.addLight(light);
}

int main(int argc, char* argv[])
{
	aten::window::init(WIDTH, HEIGHT, TITLE);

	aten::visualizer::init(
		WIDTH, HEIGHT,
		"../shader/vs.glsl",
		"../shader/fs.glsl");

	aten::vec3 lookfrom(50.0, 52.0, 295.6);
	aten::vec3 lookat(50.0, 40.8, 119.0);

	g_camera.init(
		lookfrom,
		lookat,
		aten::vec3(0, 1, 0),
		40,
		WIDTH, HEIGHT);

	makeScene();

	g_buffer.resize(WIDTH * HEIGHT);
	g_dst.resize(WIDTH * HEIGHT);

	aten::window::run(display);

	aten::window::terminate();
}