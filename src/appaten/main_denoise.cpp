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
static aten::AcceleratedScene<aten::bvh> g_scene;

static aten::StaticColorBG g_staticbg(aten::vec3(0.25, 0.25, 0.25));
static aten::envmap g_bg;
static aten::texture* g_envmap;

static aten::PathTracing g_pathtracer;
static aten::RayTracing g_raytracer;
static aten::AOVRenderer g_geotracer;

#define VFI
//#define GR

static aten::Film g_directBuffer(WIDTH, HEIGHT);

#if defined(GR)
static const int g_ratio = 2;
static aten::Film g_indirectBuffer((WIDTH / g_ratio), (HEIGHT / g_ratio));
#else
static aten::Film g_indirectBuffer(WIDTH, HEIGHT);
#endif

static aten::Film g_varIndirectBuffer(WIDTH, HEIGHT);
static aten::Film g_nml_depth_Buffer(WIDTH, HEIGHT);

#ifdef ENABLE_OMP
static uint32_t g_threadnum = 8;
#else
static uint32_t g_threadnum = 1;
#endif

static aten::PointLight virtualLight;

#if defined(VFI)
aten::VirtualFlashImage denoiser;
#elif defined(GR)
aten::GeometryRendering denoiser;
#else
aten::PracticalNoiseReduction denoiser;
#endif

void display()
{
#if defined(VFI)
    aten::Film* image = &g_directBuffer;
    aten::Film* varImage = &g_indirectBuffer;
    aten::Film* flash = &g_varIndirectBuffer;
    aten::Film* varFlash = &g_nml_depth_Buffer;

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
        virtualLight.setLe(aten::vec3(36.0, 36.0, 36.0) * real(2));
        g_pathtracer.setVirtualLight(&virtualLight, g_camera.getDir());
        g_pathtracer.render(dst, &g_scene, &g_camera);
    }

    denoiser.setParam(
        16,
        varImage->image(), 
        flash->image(),
        varFlash->image());

    aten::visualizer::render(image->image(), g_camera.needRevert());
#elif defined(GR)
    aten::Film* direct = &g_directBuffer;
    aten::Film* indirect = &g_indirectBuffer;
    aten::Film* idx = &g_varIndirectBuffer;

    {
        aten::Destination dst;
        {
            dst.width = WIDTH / g_ratio;
            dst.height = HEIGHT / g_ratio;
            dst.maxDepth = 6;
            dst.russianRouletteDepth = 3;
            dst.startDepth = 1;
            dst.sample = 16 * g_ratio * g_ratio;;
            dst.mutation = 10;
            dst.mltNum = 10;
            dst.buffer = indirect;
        }

        g_pathtracer.render(dst, &g_scene, &g_camera);
    }

    {
        aten::Destination dst;
        {
            dst.width = WIDTH;
            dst.height = HEIGHT;
            dst.maxDepth = 6;
            dst.buffer = direct;
        }

        g_raytracer.render(dst, &g_scene, &g_camera);
        //g_pathtracer.render(dst, &g_scene, &g_camera);
    }

    {
        aten::Destination dst;
        {
            dst.width = WIDTH;
            dst.height = HEIGHT;
            dst.geominfo.ids = idx;
        }

        g_geotracer.render(dst, &g_scene, &g_camera);
    }

    denoiser.setParam(
        g_ratio,
        direct->image(),
        indirect->image(),
        idx->image());

    aten::visualizer::render(g_indirectBuffer.image(), g_camera.needRevert());
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
            dst.buffer = &g_indirectBuffer;
            dst.variance = &g_varIndirectBuffer;
        }

        g_pathtracer.render(dst, &g_scene, &g_camera);
    }

    {
        aten::Destination dst;
        {
            dst.width = WIDTH;
            dst.height = HEIGHT;
            dst.maxDepth = 6;
            dst.buffer = &g_directBuffer;
        }

        g_raytracer.render(dst, &g_scene, &g_camera);
        //g_pathtracer.render(dst, &g_scene, &g_camera);
    }

    
    {
        aten::Destination dst;
        {
            dst.width = WIDTH;
            dst.height = HEIGHT;
            dst.geominfo.nml_depth = &g_nml_depth_Buffer;
            dst.geominfo.depthMax = 1000;
            dst.geominfo.needNormalize = false;
        }

        g_geotracer.render(dst, &g_scene, &g_camera);
    }

    denoiser.setBuffers(
        g_directBuffer.image(),
        g_indirectBuffer.image(),
        g_varIndirectBuffer.image(),
        g_nml_depth_Buffer.image());

    aten::visualizer::render(g_indirectBuffer.image(), g_camera.needRevert());
#endif
}

int main(int argc, char* argv[])
{
    aten::initSampler();

    aten::timer::init();
    aten::OMPUtil::setThreadNum(g_threadnum);

    aten::window::init(WIDTH, HEIGHT, TITLE);

    aten::visualizer::init(WIDTH, HEIGHT);

    //aten::BilateralFilter filter;
    //aten::NonLocalMeanFilter filter;

    aten::visualizer::addPreProc(&denoiser);
    //aten::visualizer::addPreProc(&filter);

    aten::TonemapPostProc tonemap;
    tonemap.init(
        WIDTH, HEIGHT,
        "../shader/fullscreen_vs.glsl",
        "../shader/tonemap_fs.glsl");

    aten::GammaCorrection gamma;
    gamma.init(
        WIDTH, HEIGHT,
        "../shader/fullscreen_vs.glsl",
        "../shader/gamma_fs.glsl");

    aten::Blitter blitter;
    blitter.init(
        WIDTH, HEIGHT,
        "../shader/fullscreen_vs.glsl",
        "../shader/fullscreen_fs.glsl");
    blitter.setIsRenderRGB(true);

    aten::visualizer::addPostProc(&gamma);
    //aten::visualizer::addPostProc(&tonemap);
    //aten::visualizer::addPostProc(&blitter);

    aten::vec3 lookfrom;
    aten::vec3 lookat;
    real fov;

    Scene::getCameraPosAndAt(lookfrom, lookat, fov);

#ifdef ENABLE_DOF
    g_camera.init(
        WIDTH, HEIGHT,
        lookfrom, lookat,
        aten::vec3(0, 1, 0),
        30.0,    // image sensor size
        40.0,    // distance image sensor to lens
        130.0,    // distance lens to object plane
        5.0,    // lens radius
        28.0);    // W scale
#else
    g_camera.init(
        lookfrom,
        lookat,
        aten::vec3(0, 1, 0),
        fov,
        WIDTH, HEIGHT);
#endif

    Scene::makeScene(&g_scene);

    g_scene.build();

    aten::window::run(display);

    aten::window::terminate();
}
#endif
