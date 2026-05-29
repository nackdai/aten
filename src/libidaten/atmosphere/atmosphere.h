#pragma once

#include <map>

#include "atmosphere/sky/sky_model.h"
#include "atmosphere/sky/sky_precompute_textures.h"
#include "atmosphere/rainbow/rainbow_model.h"
#include "atmosphere/rainbow/rainbow_precompute_textures.h"

#include "camera/camera.h"

#include "cuda/cudaGLresource.h"
#include "cuda/CudaSurfaceTexture.h"
#include "cuda/cudamemory.h"

namespace idaten {
    class Atmosphere {
    public:
        enum class Type {
            Sky = 1 << 0,
            Rainbow = 1 << 1,
        };

        static const std::map<int32_t, const char*> TypeMap;

        Atmosphere() = default;
        ~Atmosphere() = default;

        void Init(const aten::CameraParameter& camera);

        void PreCompute();

        void Render(
            GLuint gltex,
            const int32_t width,
            const int32_t height,
            const int32_t type,
            const float sun_zenith_angle_radians,
            const float sun_azimuth_angle_radians,
            const aten::CameraParameter& camera);

    private:
        void InitRainbow();

        void PreComputeSky();
        void PreComputeRainbow();

        struct PreComputeTexturesHost {
            // Permanent for Sky.
            CudaSurfaceTexture<float4> transmittance_texture;
            CudaSurfaceTexture<float4> irradiance_texture;
            CudaSurfaceTexture3D<float4> scattering_texture;
            CudaSurfaceTexture3D<float4> optional_single_mie_scattering_texture;

            // One shot for Sky.
            CudaSurfaceTexture<float4> delta_irradiance_texture;
            CudaSurfaceTexture3D<float4> delta_rayleigh_scattering_texture;
            CudaSurfaceTexture3D<float4> delta_mie_scattering_texture;
            CudaSurfaceTexture3D<float4> delta_scattering_density_texture;
            CudaSurfaceTexture3D<float4> delta_multiple_scattering_texture;

            // Rainbow.
            CudaSurfaceTexture<float4> transmittance_in_rain_volume_texture;
            CudaSurfaceTexture<float4> spectrum_srgb_tex;
            CudaSurfaceTexture3D<float4> droplet_radius_tex;
        } pre_compute_textures_host_;

        aten::sky::PreComputeTextureManager<idaten::SurfaceTexture, idaten::SurfaceTexture> sky_textures_;
        aten::rainbow::PreComputeTextureManager<idaten::SurfaceTexture, idaten::SurfaceTexture> rainbow_textures_;

        aten::sky::SkyModel sky_model_;
        aten::rainbow::RainbowModel rainbow_model_;

        // For rainbow.
        TypedCudaMemory<uint32_t> random_values_;

        idaten::CudaGLSurface m_glimg;
    };
}
