#pragma once

#include "atmosphere/sky/sky_model.h"
#include "atmosphere/sky/sky_params.h"
#include "atmosphere/sky/sky_precompute_textures.h"

#include "camera/camera.h"

#include "cuda/cudaGLresource.h"
#include "cuda/CudaSurfaceTexture.h"

namespace idaten::sky {
    class SkyModel : public aten::sky::SkyModel {
    public:
        SkyModel() = default;
        ~SkyModel() = default;

        void Init();

        void PreCompute();

        void Render(
            GLuint gltex,
            const int32_t width,
            const int32_t height,
            const aten::CameraParameter& camera);

    private:
        struct PreComputeTexturesHost {
            // Permanent.
            CudaSurfaceTexture<float4> transmittance_texture;
            CudaSurfaceTexture<float4> irradiance_texture;
            CudaSurfaceTexture3D<float4> scattering_texture;
            CudaSurfaceTexture3D<float4> optional_single_mie_scattering_texture;

            // One shot.
            CudaSurfaceTexture<float4> delta_irradiance_texture;
            CudaSurfaceTexture3D<float4> delta_rayleigh_scattering_texture;
            CudaSurfaceTexture3D<float4> delta_mie_scattering_texture;
            CudaSurfaceTexture3D<float4> delta_scattering_density_texture;
            CudaSurfaceTexture3D<float4> delta_multiple_scattering_texture;
        };
        PreComputeTexturesHost pre_compute_textures_host_;
        aten::sky::PreComputeTextureManager<idaten::SurfaceTexture, idaten::SurfaceTexture> textures_;

        idaten::CudaGLSurface m_glimg;
    };
}
