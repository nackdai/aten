#include "defs.h"

AT_INLINE_RELEASE AT_DEVICE_API aten::vec4 sampleTexture(
    const AT_NAME::context& ctxt,
    const int32_t texid,
    float u, float v,
    const aten::vec4& defaultValue,
    int32_t lod/*= 0*/)
{
    auto ret = defaultValue;

#ifdef __CUDACC__
    if (texid >= 0) {
        auto clr = tex2DLod<float4>(ctxt.textures[texid], u, v, lod);
        ret = aten::vec4(clr.x, clr.y, clr.z, clr.w);
    }
#endif

    return ret;
}
