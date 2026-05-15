#pragma once

#include "atmosphere/rainbow/rainbow_model.h"

#include "camera/camera.h"

#include "cuda/cudaGLresource.h"
#include "cuda/CudaSurfaceTexture.h"
#include "cuda/cudamemory.h"

namespace idaten::rainbow {
    class RainbowModel : public aten::rainbow::RainbowModel {
    public:
        RainbowModel() = default;
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

    private:
        CudaSurfaceTexture<float4> transmittance_texture_host_;
        CudaSurfaceTexture3D<float4> airy_func_texture_host_;
        CudaSurfaceTexture3D<float4> droplet_radius_texture_host_;
        CudaSurfaceTexture<float4> transmittance_in_rain_volume_texture_host_;

        idaten::SurfaceTexture transmittance_texture_;
        idaten::SurfaceTexture airy_func_texture_;
        idaten::SurfaceTexture droplet_radius_texture_;
        idaten::SurfaceTexture transmittance_in_rain_volume_texture_;

        TypedCudaMemory<uint32_t> random_values_;

        aten::aabb rain_volume_;

        idaten::CudaGLSurface m_glimg;

        TypedCudaMemory<aten::vec3> render_result_;

        // TODO
        static constexpr float intensity_rainfall_rate = 1.0F; // [mm/h]
    };
}
