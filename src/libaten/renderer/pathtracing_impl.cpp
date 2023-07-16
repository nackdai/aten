#include "renderer/pathtracing.h"
#include "camera/pinhole.h"

namespace aten
{
    void PathTracing::generate_path(
        int32_t idx,
        int32_t ix, int32_t iy,
        int32_t width, int32_t height,
        int32_t sample, uint32_t frame,
        aten::Path& paths,
        aten::ray* rays,
        const aten::CameraParameter& camera,
        const uint32_t rnd)
    {
        paths.attrib[idx].isHit = false;

        if (paths.attrib[idx].isKill) {
            paths.attrib[idx].isTerminate = true;
            return;
        }

        auto scramble = rnd * 0x1fe3434f
            * (((frame + sample) + 133 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
        paths.sampler[idx].init(
            (frame + sample) % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM),
            0,
            scramble);

        float r1 = paths.sampler[idx].nextSample();
        float r2 = paths.sampler[idx].nextSample();

        float s = (ix + r1) / (float)(camera.width);
        float t = (iy + r2) / (float)(camera.height);

        AT_NAME::CameraSampleResult camsample;
        AT_NAME::PinholeCamera::sample(&camsample, &camera, s, t);

        rays[idx] = camsample.r;

        paths.throughput[idx].throughput = aten::vec3(1);
        paths.throughput[idx].pdfb = 1.0f;
        paths.attrib[idx].isTerminate = false;
        paths.attrib[idx].isSingular = false;

        paths.contrib[idx].samples += 1;

        // TODO
        paths.contrib[idx].contrib = aten::vec3(0);
    }

    void PathTracing::shader_miss_with_envmap()
    {

    }
}
