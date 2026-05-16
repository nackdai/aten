#pragma once

#include "atmosphere/rainbow/rainbow_model.h"
#include "atmosphere/rainbow/rainbow_precompute_textures.h"

#include "camera/camera.h"

#include "cuda/cudaGLresource.h"
#include "cuda/CudaSurfaceTexture.h"
#include "cuda/cudamemory.h"

namespace idaten {
    class Atmosphere;
}

namespace idaten::rainbow {
    class RainbowModel : public aten::rainbow::RainbowModel {
    public:
        RainbowModel() {}
        ~RainbowModel() = default;

        void Init(const aten::CameraParameter& camera);

        void PreCompute();

        void Render(
            GLuint gltex,
            const int32_t width,
            const int32_t height,
            // const float sun_zenith_angle_radians,
            // const float sun_azimuth_angle_radians,
            const aten::CameraParameter& camera);

        const aten::aabb& GetRainVolume() const
        {
            return rain_volume_;
        }

    private:
        struct PreComputeTexturesHost {
            CudaSurfaceTexture<float4> transmittance_texture;
            CudaSurfaceTexture3D<float4> airy_func_tex;
            CudaSurfaceTexture3D<float4> droplet_radius_tex;
            CudaSurfaceTexture<float4> transmittance_in_rain_volume_texture;
        };

        PreComputeTexturesHost pre_compute_textures_host_;
        aten::rainbow::PreComputeTextureManager<idaten::SurfaceTexture, idaten::SurfaceTexture> textures_;

        TypedCudaMemory<uint32_t> random_values_;

        aten::aabb rain_volume_;

        idaten::CudaGLSurface m_glimg;

        // TODO
        static constexpr float intensity_rainfall_rate = 1.0F; // [mm/h]
    };
}
