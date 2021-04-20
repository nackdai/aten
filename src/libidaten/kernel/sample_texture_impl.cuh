#include "kernel/idatendefs.cuh"

AT_CUDA_INLINE AT_DEVICE_MTRL_API aten::vec4 sampleTexture(
    const int texid,
    real u, real v,
    const aten::vec4& defaultValue,
    int lod/*= 0*/)
{
    auto ret = defaultValue;

#ifdef __CUDACC__
    if (texid >= 0) {
        cudaTextureObject_t tex = (cudaTextureObject_t)texid;
        auto clr = tex2DLod<float4>(tex, u, v, lod);
        ret = aten::vec4(clr.x, clr.y, clr.z, clr.w);
    }
#endif

    return ret;
}
