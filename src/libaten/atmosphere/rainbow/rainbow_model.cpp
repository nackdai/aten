#include "atmosphere/rainbow/rainbow_model.h"

#include "atmosphere/rainbow/rainbow_compute.h"
#include "atmosphere/rainbow/rainbow_constants.h"
#include "atmosphere/rainbow/rainbow_transmittance.h"

#include "atmosphere/sky/sky_common.h"

namespace aten::rainbow {
    void RainbowModel::Init(const aten::CameraParameter& camera)
    {
        // Init 3d texture to store Airy function values.
        airy_func_tex_.init(THETA_WIDTH, WAVELENGTH_WIDTH, A_WIDTH);

        // Init 3d texture to store droplet radius based on normal distribution.
        droplet_radius_tex_.init(DROPLET_RADIUS_TEX_SIZE, DROPLET_RADIUS_TEX_SIZE, DROPLET_RADIUS_TEX_SIZE);

        transmittance_texture_.init(
            aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
            aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
            3);

        transmittance_in_rain_volume_texture_.init(
            aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
            aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
            3);

        // Set rain volume box.
        {
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

            rain_volume_.init(
                rain_volume_min,
                rain_volume_max);
        }

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

        void ComputeTransmittanceInRainVolume(
            int32_t x, int32_t y,
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::aabb& rain_volume,
            const float extinction,
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

            rainbow::GetRMuFromTransmittanceTextureUv(
                atmosphere, rain_volume,
                frag_coord / TRANSMITTANCE_TEXTURE_SIZE,
                r, mu);

            const auto optical_depth = rainbow::ComputeOpticalDepthBasedOnAabbCoveredSphere(
                atmosphere, rain_volume,
                r, mu);

            const auto transmittance = aten::exp(-extinction * optical_depth);
            AT_ASSERT(transmittance <= 1.0F);

            sky::WriteTexture2D(
                transmittance_texture,
                aten::vec3(transmittance),
                x, y);
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

            const float extinction = ComputeExtinctionInRain(intensity_rainfall_rate);

            // Precompute transmittance in rain volume.
#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
            for (int32_t y = 0; y < aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT; y++) {
                for (int32_t x = 0; x < aten::sky::TRANSMITTANCE_TEXTURE_WIDTH; x++) {
                    ComputeTransmittanceInRainVolume(
                        x, y,
                        atmosphere_, rain_volume_, extinction,
                        transmittance_in_rain_volume_texture_);
                }
            }

            // TODO
            const auto rnd = aten::getRandom(0);
            const auto frame = 0;
            const auto scramble = rnd * 0x1fe3434f * (((frame + rnd) + 133 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
            aten::CMJ sampler;
            sampler.init(
                (frame + rnd) % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM),
                0,
                scramble);

            constexpr auto mu = 0.5_mm;

            // NOTE:
            // Sigma for 95% of normal distribution is precisely 1.96.
            // We can often see 2.0. But, it's approximation. It's not correct mathematically.
            // In this case, the target range is 0.3 - 0.7F.
            constexpr auto sigma = 0.2_mm / 1.96F;

            // Precompute droplet radius volume.
#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
            for (int32_t z = 0; z < DROPLET_RADIUS_TEX_SIZE; z++) {
                for (int32_t y = 0; y < DROPLET_RADIUS_TEX_SIZE; y++) {
                    for (int32_t x = 0; x < DROPLET_RADIUS_TEX_SIZE; x++) {
                        auto u = 0.0F;
                        #pragma omp critical
                        {
                            u = sampler.nextSample();
                        }

                        const auto droplet_radius = ComputeInverseNormalDistributionCDF(u, mu, sigma);
                        droplet_radius_tex_.SetByXYZ(droplet_radius, x, y, z);
                    }
                }
            }
        }
    }

//#pragma optimize("", off)


    namespace {
        aten::vec3 RenderRainbow(
            aten::CMJ& sampler,
            const int32_t x, const int32_t y,
            const aten::CameraParameter& camera,
            const sky::AtmosphereParameters& atmosphere,
            const aten::sky::texture2d& transmittance_texture,
            const aten::sky::texture2d& transmittance_in_rain_volume_texture,
            const aten::sky::texture3d& droplet_radius_tex,
            const aten::vec3& sun_direction,
            const aten::vec3& earth_center, // [km]
            const aten::aabb& rain_volume,  // [km x km x km]
            const float intensity_rainfall_rate,    // [mm/h]
            const float extinction,
            const aten::texture3d& airy_func_res_tex,
            const aten::vec3& sun_radiance_to_luminance,
            const aten::vec3& white_point
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

            //AT_PRINTF("np.array([%f, %f, %f]),\n", view_dir.x, view_dir.y, view_dir.z);

            aten::vec3 rainbow_radiance {
                AdvanceRainVolumeIntegral(
                    sampler,
                    atmosphere,
                    transmittance_texture,
                    transmittance_in_rain_volume_texture,
                    droplet_radius_tex,
                    sun_direction,
                    earth_center, // [km]
                    camera_pos,   // [km]
                    view_dir,
                    rain_volume,  // [km x km x km]
                    intensity_rainfall_rate,    // [mm/h]
                    extinction,
                    airy_func_res_tex)
            };

            //AT_PRINTF("%d, ", y);
            //AT_PRINTF("%f, %f, %f, ", rainbow_radiance.x, rainbow_radiance.y, rainbow_radiance.z);

            rainbow_radiance *= sun_radiance_to_luminance;

            //AT_PRINTF("%d, ", y);
            //AT_PRINTF("%f, %f, %f, ", rainbow_radiance.x, rainbow_radiance.y, rainbow_radiance.z);

            aten::vec3 color{
                aten::vec3(1.0F) - aten::exp(-rainbow_radiance / white_point * aten::sky::EXPOSURE)
            };

            //AT_PRINTF("%d, ", y);
            //AT_PRINTF("%f, %f, %f,", color.x, color.y, color.z);
            //AT_PRINTF("\n");

            return color;
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

        const float extinction = ComputeExtinctionInRain(intensity_rainfall_rate);

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp parallel
#endif
        {
#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
//#pragma omp for schedule(dynamic, 1)
#endif
            for (int32_t y = 0; y < height; y++) {
                //const auto x = 128;
                for (int32_t x = 0; x < width; x++)
                //for (int32_t x = 112; x < 144; x++)
                {
                    const auto id = y * width + x;
                    const auto rnd = aten::getRandom(id);
                    const auto frame = 0;
                    const auto sample = 0;

                    auto scramble = rnd * 0x1fe3434f
                        * (((frame + sample) + 133 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));

                    aten::CMJ sampler;
                    sampler.init(
                        (frame + sample) % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM),
                        0,
                        scramble);

                    const auto color {
                        RenderRainbow(
                            sampler,
                            x, y,
                            camera,
                            atmosphere_,
                            transmittance_texture_,
                            transmittance_in_rain_volume_texture_,
                            droplet_radius_tex_,
                            sun_direction,
                            earth_center,
                            rain_volume_,
                            intensity_rainfall_rate,
                            extinction,
                            airy_func_tex_,
                            sun_radiance_to_luminance_, white_point_)
                    };

                    const auto l = length(color);

                    //if (x == 0)
                    if (y == 103)
                    {
                        //AT_PRINTF("[%d, %d] : %f, %f, %f\n", x, y, color.r, color.g, color.b);
                    }
                    dst.put(x, y, color);
                }
            }
        }
    }
}
