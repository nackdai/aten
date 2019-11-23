#include "asvgf/asvgf.h"

#include "kernel/StreamCompaction.h"

#include "kernel/context.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/intersect.cuh"
#include "kernel/accelerator.cuh"
#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

__global__ void genPathASVGF(
    idaten::TileDomain tileDomain,
    bool isFillAOV,
    idaten::SVGFPathTracing::Path* paths,
    aten::ray* rays,
    int width, int height,
    int maxBounces,
    unsigned int frame,
    const aten::CameraParameter* __restrict__ camera,
    cudaTextureObject_t blueNoise,
    int blueNoiseResW, int blueNoiseResH, int blueNoiseLayerNum,
    const unsigned int* __restrict__ random)
{
    auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    const auto idx = getIdx(ix, iy, width);

    paths->attrib[idx].isHit = false;

    if (paths->attrib[idx].isKill) {
        paths->attrib[idx].isTerminate = true;
        return;
    }

#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
    auto scramble = random[idx] * 0x1fe3434f;
    paths->sampler[idx].init(frame, 0, scramble, samplerValues);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
    auto rnd = random[idx];
    auto scramble = rnd * 0x1fe3434f * ((frame + 133 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
    paths->sampler[idx].init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 0, scramble);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_BLUENOISE
    auto seed = random[idx];
    paths->sampler[idx].init(
        //seed,
        ix, iy, frame,
        maxBounces,
        idaten::SVGFPathTracing::ShadowRayNum,
        blueNoiseResW, blueNoiseResH, blueNoiseLayerNum,
        blueNoise);
#endif

    float r1 = paths->sampler[idx].nextSample();
    float r2 = paths->sampler[idx].nextSample();

    if (isFillAOV) {
        r1 = r2 = 0.5f;
    }

    ix += tileDomain.x;
    iy += tileDomain.y;

    float s = (ix + r1) / (float)(camera->width);
    float t = (iy + r2) / (float)(camera->height);

    AT_NAME::CameraSampleResult camsample;
    AT_NAME::PinholeCamera::sample(&camsample, camera, s, t);

    rays[idx] = camsample.r;

    paths->throughput[idx].throughput = aten::vec3(1);
    paths->throughput[idx].pdfb = 0.0f;
    paths->attrib[idx].isTerminate = false;
    paths->attrib[idx].isSingular = false;

    paths->contrib[idx].samples += 1;
}

namespace idaten
{
    void AdvancedSVGFPathTracing::onGenPath(
        int maxBounce,
        int seed,
        cudaTextureObject_t texVtxPos,
        cudaTextureObject_t texVtxNml)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        bool isFillAOV = m_mode == Mode::AOVar;

        auto blueNoise = m_bluenoise.bind();
        auto blueNoiseResW = m_bluenoise.getWidth();
        auto blueNoiseResH = m_bluenoise.getHeight();
        auto blueNoiseLayerNum = m_bluenoise.getLayerNum();

        int curRngSeed = (m_frame & 0x01);

        genPathASVGF << <grid, block, 0, m_stream >> > (
            m_tileDomain,
            isFillAOV,
            m_paths.ptr(),
            m_rays.ptr(),
            m_tileDomain.w, m_tileDomain.h,
            maxBounce,
            m_frame,
            m_cam.ptr(),
            blueNoise,
            blueNoiseResW, blueNoiseResH, blueNoiseLayerNum,
            m_rngSeed[curRngSeed].ptr());

        checkCudaKernel(genPath);
    }
}
