#include "atmosphere/atmosphere.h"

#include "atmosphere/rainbow/rainbow_compute.h"
#include "atmosphere/rainbow/rainbow_render.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

namespace idaten {
    void Atmosphere::Init(const aten::CameraParameter& camera)
    {
        sky_textures_.Init(pre_compute_textures_host_);
        rainbow_textures_.Init(pre_compute_textures_host_);

        aten::sky::SkyModel::InitParameters(sky_model_);

        InitRainbow();

        // Set rain volume box.
        {
            // TODO
            constexpr aten::Length RainVolumeWidth = 4.0_km;
            constexpr aten::Length RainVolumeHeight = 4.0_km;
            constexpr aten::Length RainVolumeDepth = 4.0_km;

            // TODO
            const auto& camera_pos = camera.origin;

            aten::vec3 rain_volume_min{
                camera_pos.x - RainVolumeWidth.as(aten::MeterUnit::km) * 0.5f,
                0.0F,
                camera_pos.z - 1.0F - RainVolumeDepth.as(aten::MeterUnit::km),
            };
            aten::vec3 rain_volume_max{
                camera_pos.x + RainVolumeWidth.as(aten::MeterUnit::km) * 0.5f,
                rain_volume_min.y + RainVolumeHeight.as(aten::MeterUnit::km),
                camera_pos.z - 1.0F,
            };

            rainbow_model_.rain_volume_.init(
                rain_volume_min,
                rain_volume_max);
        }
    }

    void Atmosphere::PreCompute()
    {
        PreComputeSky();
        PreComputeRainbow();
    }

    __device__ inline bool WillRenderAtmosphere(
        const int32_t type,
        const Atmosphere::Type render_type)
    {
        return (type & static_cast<int32_t>(render_type)) > 0;
    }

    __global__ void RenderAtmosphere(
        cudaSurfaceObject_t dst,
        int32_t width, int32_t height,
        const int32_t type,
        const aten::CameraParameter camera,
        const aten::sky::AtmosphereParameters atmosphere,
        const aten::sky::PreComputeTextures sky_textures,
        const aten::rainbow::PreComputeTextureManager<idaten::SurfaceTexture, idaten::SurfaceTexture> rainbow_textures,
        const aten::aabb rain_volume,
        const float intensity_rainfall_rate,    // [mm/h]
        const uint32_t* random_values,
        const aten::vec3 sun_radiance_to_luminance,
        const aten::vec3 sky_radiance_to_luminance,
        const aten::vec3 sun_direction,
        const aten::vec3 earth_center,
        const float sun_size,
        const aten::vec3 white_point)
    {
        const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        const float s = x / static_cast<float>(camera.width);
        const float t = y / static_cast<float>(camera.height);

        // TODO
        // Pinhole?
        AT_NAME::CameraSampleResult camsample;
        AT_NAME::PinholeCamera::sample(&camsample, &camera, s, t);

        const auto camera_pos{ camsample.r.org };
        const auto view_dir{ camsample.r.dir };

        const auto will_render_sky = WillRenderAtmosphere(type, Atmosphere::Type::Sky);
        const auto will_render_rainbow = WillRenderAtmosphere(type, Atmosphere::Type::Rainbow);

        aten::vec3 atmosphere_color{ 0.0F };

        if (will_render_rainbow) {
            const auto id = y * width + x;

            const auto rnd = random_values[id];
            const auto frame = 0;
            const auto scramble = rnd * 0x1fe3434f * (((frame + rnd) + 133 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
            aten::CMJ sampler;
            sampler.init(
                (frame + rnd) % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM),
                0,
                scramble);

            aten::vec3 rainbow_radiance{
                aten::rainbow::AdvanceRainVolumeIntegral(
                    sampler,
                    atmosphere,
                    rainbow_textures.transmittance_texture,
                    rainbow_textures.transmittance_in_rain_volume_texture,
                    rainbow_textures.droplet_radius_tex,
                    sun_direction,
                    earth_center, // [km]
                    camera_pos,   // [km]
                    view_dir,
                    rain_volume,  // [km x km x km]
                    intensity_rainfall_rate,    // [mm/h]
                    rainbow_textures.spectrum_srgb_tex)
            };

            rainbow_radiance = aten::vmax(rainbow_radiance, 0.0F);
            atmosphere_color += rainbow_radiance * sun_radiance_to_luminance;
        }

        if (will_render_sky) {
            auto sky_luminance{
                aten::sky::RenderSky(
                    x, y,
                    camera,
                    atmosphere, sky_textures,
                    sun_radiance_to_luminance, sky_radiance_to_luminance,
                    sun_direction,
                    earth_center,
                    sun_size)
            };

            // TODO:
            // The rain volume is a virtual medium used only to simulate the rainbow.
            // It does not currently represent visible rain or a full rain simulation.
            // Decide whether the same medium should also attenuate the background sky.
            // If both looks are useful, expose this as a rendering option.
#if 0
            if (will_render_rainbow) {
                float t0, t1;
                aten::tie(t0, t1) = rain_volume.GetHitT(aten::ray(camera_pos, view_dir), AT_MATH_EPSILON, AT_MATH_INF);

                const auto is_hit = t0 <= t1;

                if (is_hit) {
                    const auto near_boundary_point = camera_pos + view_dir * t0;
                    const auto d = t1 - t0;

                    const auto transmittance_through_rain_volume{
                        aten::rainbow::GetTransmittanceInRainVolume(
                            atmosphere, rain_volume, earth_center,
                            rainbow_textures.transmittance_in_rain_volume_texture,
                            near_boundary_point, view_dir, d)
                    };

                    sky_luminance *= transmittance_through_rain_volume;
                }
            }
#endif

            atmosphere_color += sky_luminance;
        }

        // TODO
        // Tone mapping.
        // white point (RGB=1.0（白））に対する比率の負値のexponential -> 強い値ほど減衰（ゼロに近い）.
        // それを 1.0 から引くことで、結果強い値が大きくなる.
        // exposure は全体の明るさを調整するための係数.
        aten::vec3 color{
            aten::vec3(1.0F) - aten::exp(-atmosphere_color / white_point * aten::sky::EXPOSURE)
        };

        surf2Dwrite(
            make_float4(color.x, color.y, color.z, 1.0F),
            dst,
            x * sizeof(float4), y,
            cudaBoundaryModeTrap);
    }

    void Atmosphere::Render(
        GLuint gltex,
        const int32_t width,
        const int32_t height,
        const int32_t type,
        const float sun_zenith_angle_radians,
        const float sun_azimuth_angle_radians,
        const aten::CameraParameter& camera)
    {
        if (!m_glimg.IsValid()) {
            m_glimg.init(gltex, CudaGLRscRegisterType::ReadWrite);
        }

        const dim3 thread_per_block{ 16, 16 };
        const dim3 block_per_grid{
            (width + thread_per_block.x - 1) / thread_per_block.x,
            (height + thread_per_block.y - 1) / thread_per_block.y
        };

#if 1
        aten::vec3 sun_direction{
            aten::sin(sun_zenith_angle_radians) * aten::cos(sun_azimuth_angle_radians),
            aten::cos(sun_zenith_angle_radians),
            aten::sin(sun_zenith_angle_radians) * aten::sin(sun_azimuth_angle_radians)
        };
#else
        // For debug.
        constexpr auto sun_angle = aten::Deg2Rad(20.0F);
        aten::vec3 sun_direction{
            0.0F,
            aten::sin(sun_angle),
            aten::cos(sun_angle),
        };
#endif
        sun_direction = normalize(sun_direction);

        const aten::vec3 earth_center{
            0.0F,
            -aten::sky::BottomRadius.as(aten::MeterUnit::km),
            0.0F,
        };

        const auto sun_size = aten::cos(aten::sky::SunAngularRadius);

        CudaGLResourceMapper<decltype(m_glimg)> rscmap(m_glimg);
        auto output_surface = m_glimg.bind();

        RenderAtmosphere << <block_per_grid, thread_per_block >> > (
            output_surface,
            width, height,
            type,
            camera,
            sky_model_.atmosphere_,
            sky_textures_,
            rainbow_textures_,
            rainbow_model_.rain_volume_,
            aten::rainbow::RainbowModel::intensity_rainfall_rate,
            random_values_.data(),
            sky_model_.sun_radiance_to_luminance_,
            sky_model_.sky_radiance_to_luminance_,
            sun_direction,
            earth_center,
            sun_size,
            sky_model_.white_point_
        );
        checkCudaKernel(RenderAtmosphere);

        m_glimg.unbind();
    }

    const std::map<int32_t, const char*> Atmosphere::TypeMap{
        { static_cast<int32_t>(Type::Sky), "Sky" },
        { static_cast<int32_t>(Type::Rainbow), "Rainbow" },
    };
}
