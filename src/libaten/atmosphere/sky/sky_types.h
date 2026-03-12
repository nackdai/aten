#pragma once

#include "atmosphere/sky/sky_precompute_textures.h"

#ifdef __CUDACC__
#include "cuda/cudadefs.h"
#include "cuda/CudaSurfaceTexture.h"
#else
#include "image/texture.h"
#include "image/texture_3d.h"
#endif

namespace aten::sky {
#ifdef __CUDACC__
    using texture2d = idaten::SurfaceTexture;
    using texture3d = idaten::SurfaceTexture;
    using PreComputeTextures = PreComputeTextureManager<idaten::SurfaceTexture, idaten::SurfaceTexture>;
#else
    using texture2d = aten::texture;
    using texture3d = aten::texture3d;
    using PreComputeTextures = PreComputeTextureManager<aten::texture, aten::texture3d>;
#endif
}
