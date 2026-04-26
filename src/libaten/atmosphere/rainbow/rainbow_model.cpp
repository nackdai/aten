#include "atmosphere/rainbow/rainbow_model.h"

#include "atmosphere/rainbow/rainbow_compute.h"
#include "atmosphere/rainbow/rainbow_constants.h"

#include "atmosphere/sky/sky_common.h"

namespace aten::rainbow {
    void RainbowModel::Init()
    {
        // Init 3d texture to store Airy function values.
        airy_func_tex_.init(THETA_WIDTH, WAVELENGTH_WIDTH, A_WIDTH);

        transmittance_texture_.init(
            aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
            aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
            3);

        SkyModel::InitParameters();
    }

    // TODO
    // Unify to sky model side code.
    namespace {
        // Transmittance を計算.
        void ComputeTransmittanceToTopAtmosphereBoundaryTexture(
            const aten::sky::AtmosphereParameters& atmosphere,
            const int32_t x, const int32_t y,
            aten::texture& transmittance_texture)
        {
            static const aten::vec2 TRANSMITTANCE_TEXTURE_SIZE{
                aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
                aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
            };

            const aten::vec2 frag_coord{
                static_cast<float>(x) + 0.5F,
                static_cast<float>(y) + 0.5F,
            };

            float r;
            float mu;

            aten::sky::GetRMuFromTransmittanceTextureUv(
                atmosphere, frag_coord / TRANSMITTANCE_TEXTURE_SIZE, r, mu);

            // rとmuから上端境界までのtransmittanceを計算する.
            const auto transmittance{
                aten::sky::transmittance::ComputeTransmittanceToTopAtmosphereBoundary(
                    atmosphere,
                    r, mu)
            };

            aten::sky::WriteTexture2D(transmittance_texture, transmittance, x, y);
        }
    }

    void RainbowModel::PreCompute()
    {
#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp parallel
#endif
        {
            // Transmittance を計算.
#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
            for (int32_t y = 0; y < aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT; y++) {
                for (int32_t x = 0; x < aten::sky::TRANSMITTANCE_TEXTURE_WIDTH; x++) {
                    ComputeTransmittanceToTopAtmosphereBoundaryTexture(
                        atmosphere_,
                        x, y,
                        transmittance_texture_);
                }
            }

            // Precompute Airy function table.
#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
            for (int32_t z = 0; z < A_WIDTH; z++) {
                for (int32_t y = 0; y < WAVELENGTH_WIDTH; y++) {
                    for (int32_t x = 0; x < THETA_WIDTH; x++) {
                        const auto intensity = ComputeAiryFunction(x, y, z);
                        airy_func_tex_.SetByXYZ(vec4(intensity), x, y, z);
                    }
                }
            }
        }
    }

//#pragma optimize("", off)


    namespace {
        aten::vec3 RenderRainbow(
            const int32_t x, const int32_t y,
            const aten::CameraParameter& camera,
            const sky::AtmosphereParameters& atmosphere,
            const aten::sky::texture2d& transmittance_texture,
            const aten::vec3& sun_direction,
            const aten::vec3& earth_center, // [km]
            const aten::aabb& rain_volume,  // [km x km x km]
            const float droplet_diameter,   // [m]
            const float intensity_rainfall_rate,    // [mm/h]
            const aten::texture3d& airy_func_res_tex,
            const aten::vec3& sun_radiance_to_luminance
        )
        {
            const float s = x / static_cast<float>(camera.width);
            const float t = y / static_cast<float>(camera.height);

            // TODO
            // Pinhole?
            AT_NAME::CameraSampleResult camsample;
            AT_NAME::PinholeCamera::sample(&camsample, &camera, s, t);

            const auto camera_pos{ camsample.r.org };
            const auto view_dir{ camsample.r.dir };

            aten::vec3 rainbow_radiance {
                AdvanceRainVolumeIntegral(
                    atmosphere,
                    transmittance_texture,
                    sun_direction,
                    earth_center, // [km]
                    camera_pos,   // [km]
                    view_dir,
                    rain_volume,  // [km x km x km]
                    droplet_diameter,   // [m]
                    intensity_rainfall_rate,    // [mm/h]
                    airy_func_res_tex)
            };

            rainbow_radiance *= sun_radiance_to_luminance;

            return rainbow_radiance;
        }
    }

    // NOTE
    // camera parameters has to be specified based on km unit.
    void RainbowModel::Render(
        const int32_t width,
        const int32_t height,
        const aten::CameraParameter& camera,
        Film& dst)
    {
        constexpr auto sun_angle = Deg2Rad(20.0F);

        // TODO
        aten::vec3 sun_direction{
            0.0F,
            aten::sin(sun_angle),
            aten::cos(sun_angle),
        };
        sun_direction = normalize(sun_direction);

        const aten::vec3 earth_center{
            0.0F,
            -sky::BottomRadius.as(MeterUnit::km),
            0.0F,
        };

        // TODO
        constexpr Length RainVolumeWidth = 4.0_km;
        constexpr Length RainVolumeHeight = 4.0_km;
        constexpr Length RainVolumeDepth = 4.0_km;

        // TODO
        const auto& camera_pos = camera.origin;

        aten::vec3 rain_volume_min{
            camera_pos.x - RainVolumeWidth.as(MeterUnit::km) * 0.5f,
            0.0F,
            camera_pos.z - 1.0F - RainVolumeDepth.as(MeterUnit::km),
        };
        aten::vec3 rain_volume_max{
            camera_pos.x + RainVolumeWidth.as(MeterUnit::km) * 0.5f,
            rain_volume_min.y + RainVolumeHeight.as(MeterUnit::km),
            camera_pos.z - 1.0F,
        };

        aten::aabb rain_volume{
            rain_volume_min,
            rain_volume_max,
        };

        // TODO
        constexpr Length droplet_diameter = 1.0_mm;
        constexpr float intensity_rainfall_rate = 1.0F; // [mm/h]

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp parallel
#endif
        {
#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
//#pragma omp for
#pragma omp for schedule(dynamic, 1)
#endif
            for (int32_t y = 0; y < height; y++) {
                for (int32_t x = 0; x < width; x++) {
                    const auto color {
                        RenderRainbow(
                            x, y,
                            camera,
                            atmosphere_,
                            transmittance_texture_,
                            sun_direction,
                            earth_center,
                            rain_volume,
                            droplet_diameter,
                            intensity_rainfall_rate,
                            airy_func_tex_,
                            sun_radiance_to_luminance_)
                    };

                    if (x == 0) {
                        AT_PRINTF("[%d, %d] : %f, %f, %f\n", x, y, color.r, color.g, color.b);
                    }
                    dst.put(x, y, color * 100.0F);
                }
            }
        }
    }
}
