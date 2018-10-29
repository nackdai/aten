#if 1
#include <vector>
#include "aten.h"
#include "atenscene.h"
#include "../common/scenedefs.h"

static int WIDTH = 512;
static int HEIGHT = 512;
static const char* TITLE = "app";

//#define ENABLE_EVERY_FRAME_SC
//#define ENABLE_DOF

#ifdef ENABLE_DOF
static aten::ThinLensCamera g_camera;
#else
static aten::PinholeCamera g_camera;
#endif

static aten::AcceleratedScene<aten::sbvh> g_scene;
static aten::context g_ctxt;

static aten::StaticColorBG g_staticbg(aten::vec3(0.25, 0.25, 0.25));
static aten::envmap g_bg;
static aten::texture* g_envmap;

//static aten::RayTracing g_tracer;
static aten::PathTracing g_tracer;
//static aten::BDPT g_tracer;
//static aten::SortedPathTracing g_tracer;
//static aten::ERPT g_tracer;
//static aten::PSSMLT g_tracer;
//static aten::GeometryInfoRenderer g_tracer;

static aten::visualizer* g_visualizer;

static aten::FilmProgressive g_buffer(WIDTH, HEIGHT);
//static aten::Film g_buffer(WIDTH, HEIGHT);

static aten::RasterizeRenderer g_rasterizerAABB;

static bool isExportedHdr = false;

#ifdef ENABLE_OMP
static uint32_t g_threadnum = 8;
#else
static uint32_t g_threadnum = 1;
#endif

static uint32_t g_frameNo = 0;
static float g_avgElapsed = 0.0f;

void update()
{
    static float y = 0.0f;
    static float d = -0.1f;

    auto obj = getMovableObj();

    if (obj) {
        auto t = obj->getTrans();

        if (y >= 0.0f) {
            d = -0.1f;
        }
        else if (y <= -0.5f) {
            d = 0.1f;

        }

        y += d;
        t.y += d;

        obj->setTrans(t);
        obj->update();

        auto accel = g_scene.getAccel();
        accel->update(g_ctxt);
    }
}

void display(aten::window* wnd)
{
    update();

    g_camera.update();

    aten::Destination dst;
    {
        dst.width = WIDTH;
        dst.height = HEIGHT;
        dst.maxDepth = 5;
        dst.russianRouletteDepth = 3;
        dst.startDepth = 0;
        dst.sample = 1;
        dst.mutation = 10;
        dst.mltNum = 10;
        dst.buffer = &g_buffer;
    }

    dst.geominfo.albedo_vis = &g_buffer;
    dst.geominfo.depthMax = 1000;

    aten::timer timer;
    timer.begin();

    // Trace rays.
    g_tracer.render(g_ctxt, dst, &g_scene, &g_camera);

    auto elapsed = timer.end();

    g_avgElapsed = g_avgElapsed * g_frameNo + elapsed;
    g_avgElapsed /= (g_frameNo + 1);

    AT_PRINTF("Elapsed %f[ms] / Avg %f[ms]\n", elapsed, g_avgElapsed);

    if (!isExportedHdr) {
        isExportedHdr = true;

        // Export to hdr format.
        aten::HDRExporter::save(
            "result.hdr",
            g_buffer.image(),
            WIDTH, HEIGHT);
    }

    g_visualizer->render(g_buffer.image(), g_camera.needRevert());

    g_rasterizerAABB.drawAABB(
        &g_camera,
        g_scene.getAccel());

#ifdef ENABLE_EVERY_FRAME_SC
    {
        static char tmp[1024];
        sprintf(tmp, "sc_%d.png\0", g_frameNo);

        g_visualizer->takeScreenshot(tmp);
    }
#endif
    g_frameNo++;
}

int main(int argc, char* argv[])
{
    aten::initSampler(WIDTH, HEIGHT, 0, true);

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
    aten::OMPUtil::setThreadNum(g_threadnum);

    aten::window::init(WIDTH, HEIGHT, TITLE, display);

    g_visualizer = aten::visualizer::init(WIDTH, HEIGHT);

    aten::Blitter blitter;
    blitter.init(
        WIDTH, HEIGHT,
        "../shader/fullscreen_vs.glsl",
        "../shader/fullscreen_fs.glsl");
    blitter.setIsRenderRGB(true);

    aten::TonemapPostProc tonemap;
    tonemap.init(
        WIDTH, HEIGHT,
        "../shader/fullscreen_vs.glsl",
        "../shader/tonemap_fs.glsl");

    aten::NonLocalMeanFilterShader nlmshd;
    nlmshd.init(
        WIDTH, HEIGHT,
        "../shader/fullscreen_vs.glsl",
        "../shader/nlm_fs.glsl");

    aten::BilateralFilterShader bishd;
    bishd.init(
        WIDTH, HEIGHT,
        "../shader/fullscreen_vs.glsl",
        "../shader/bilateral_fs.glsl");

    aten::BloomEffect bloom;
    bloom.init(
        WIDTH, HEIGHT,
        aten::rgba32f, aten::rgba32f,
        "../shader/fullscreen_vs.glsl",
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
        "../shader/fullscreen_vs.glsl",
        "../shader/gamma_fs.glsl");

    //aten::visualizer::addPostProc(&bishd);
    //aten::visualizer::addPostProc(&blitter);
    g_visualizer->addPostProc(&gamma);
    //aten::visualizer::addPostProc(&tonemap);
    //aten::visualizer::addPostProc(&bloom);

    g_rasterizerAABB.init(
        WIDTH, HEIGHT,
        "../shader/simple3d_vs.glsl",
        "../shader/simple3d_fs.glsl");

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

    Scene::makeScene(g_ctxt, &g_scene);

    g_scene.build(g_ctxt);
    //g_scene.getAccel()->computeVoxelLodErrorMetric(HEIGHT, fov, 4);

    //g_tracer.setVirtualLight(g_camera.getPos(), g_camera.getDir(), aten::vec3(36.0, 36.0, 36.0)* 2);

    g_envmap = aten::ImageLoader::load("../../asset/envmap/studio015.hdr", g_ctxt);
    //g_envmap = aten::ImageLoader::load("../../asset/envmap/harbor.hdr");
    g_bg.init(g_envmap);

    aten::ImageBasedLight ibl(&g_bg);
    g_scene.addImageBasedLight(&ibl);

    //g_tracer.setBG(&g_staticbg);

    //aten::NonLocalMeanFilter filter;
    //aten::BilateralFilter filter;
    //aten::visualizer::addPreProc(&filter);

    aten::window::run();

    aten::window::terminate();
}
#endif
