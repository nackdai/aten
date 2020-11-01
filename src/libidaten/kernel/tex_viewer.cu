#include "kernel/renderer.h"
#include "kernel/pt_common.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

__global__ void textureViewer(
    uint32_t texIdx,
    int width, int height,
    cudaTextureObject_t* textures,
    cudaSurfaceObject_t outSurface)
{
    const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    float u = ix / (float)width;
    float v = iy / (float)height;

    auto texclr = AT_NAME::sampleTexture(textures[texIdx], u, v, aten::vec3(1.0f));

    surf2Dwrite(
        make_float4(texclr.r, texclr.g, texclr.b, 1.0f),
        outSurface,
        ix * sizeof(float4), iy,
        cudaBoundaryModeTrap);
}

namespace idaten
{
    void Renderer::viewTextures(
        uint32_t idx,
        int screenWidth, int screenHeight)
    {
        m_glimg.map();
        auto outputSurf = m_glimg.bind();

        if (!m_texRsc.empty()) {
            std::vector<cudaTextureObject_t> tmp;
            for (int i = 0; i < m_texRsc.size(); i++) {
                auto cudaTex = m_texRsc[i].bind();
                tmp.push_back(cudaTex);
            }
            m_tex.writeByNum(&tmp[0], (uint32_t)tmp.size());
        }

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (screenWidth + block.x - 1) / block.x,
            (screenHeight + block.y - 1) / block.y);

        textureViewer << <grid, block >> > (
            idx,
            screenWidth, screenHeight,
            m_tex.ptr(),
            outputSurf);

        for (int i = 0; i < m_texRsc.size(); i++) {
            m_texRsc[i].unbind();
        }

        m_glimg.unbind();
        m_glimg.unmap();
    }
}
