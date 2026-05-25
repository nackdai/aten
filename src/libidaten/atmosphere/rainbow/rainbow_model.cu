#include "rainbow_model_device.h"

#include <random>
#include <algorithm>

#include "atmosphere/rainbow/rainbow_constants.h"
#include "atmosphere/rainbow/rainbow_compute.h"
#include "atmosphere/rainbow/rainbow_render.h"
#include "atmosphere/sky/sky_compute.h"

#include "atmosphere/atmosphere.h"

#include "sampler/cmj.h"

namespace idaten::rainbow {
    static void InitRandom(TypedCudaMemory<uint32_t>& random_values)
    {
        // TODO.
        // Create random values to pick the value from normal distribution.
        const auto random_store_size = aten::rainbow::DROPLET_RADIUS_TEX_SIZE
            * aten::rainbow::DROPLET_RADIUS_TEX_SIZE
            * aten::rainbow::DROPLET_RADIUS_TEX_SIZE;
        const auto seed = 0;
        std::vector<uint32_t> random(random_store_size);
        std::mt19937 rand_src(seed);
        std::generate(random.begin(), random.end(), rand_src);
        random_values.resize(random_store_size);
        random_values.writeFromHostToDeviceByNum(random.data(), random_store_size);
    }

    void RainbowModel::Init(const aten::CameraParameter& camera)
    {
        textures_.Init(pre_compute_textures_host_);

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

            rain_volume_.init(
                rain_volume_min,
                rain_volume_max);
        }

        // TODO.
        // Create random values to pick the value from normal distribution.
        const auto random_store_size = aten::rainbow::DROPLET_RADIUS_TEX_SIZE
            * aten::rainbow::DROPLET_RADIUS_TEX_SIZE
            * aten::rainbow::DROPLET_RADIUS_TEX_SIZE;
        const auto seed = 0;
        std::vector<uint32_t> random(random_store_size);
        std::mt19937 rand_src(seed);
        std::generate(random.begin(), random.end(), rand_src);
        random_values_.resize(random_store_size);
        random_values_.writeFromHostToDeviceByNum(random.data(), random_store_size);

        SkyModel::InitParameters(*this);
    }

    namespace {
        // Transmittance を計算.
        __global__ void ComputeTransmittanceToTopAtmosphereBoundaryTexture(
            const aten::sky::AtmosphereParameters atmosphere,
            idaten::SurfaceTexture transmittance_texture)
        {
            const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= aten::sky::TRANSMITTANCE_TEXTURE_WIDTH || y >= aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT) {
                return;
            }

            const aten::vec2 TRANSMITTANCE_TEXTURE_SIZE{
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

        __global__ void ComputeTransmittanceInRainVolumeTexture(
            const aten::sky::AtmosphereParameters atmosphere,
            idaten::SurfaceTexture transmittance_in_rain_volume_texture,
            aten::aabb rain_volume,
            float extinction)
        {
            const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= aten::sky::TRANSMITTANCE_TEXTURE_WIDTH || y >= aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT) {
                return;
            }

            const aten::vec2 TRANSMITTANCE_TEXTURE_SIZE{
                aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
                aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
            };

            const aten::vec2 frag_coord{
                static_cast<float>(x) + 0.5F,
                static_cast<float>(y) + 0.5F,
            };

            float r;
            float mu;

            aten::rainbow::GetRMuFromTransmittanceTextureUv(
                atmosphere, rain_volume,
                frag_coord / TRANSMITTANCE_TEXTURE_SIZE,
                r, mu);

            const auto optical_depth = aten::rainbow::ComputeOpticalDepthBasedOnAabbCoveredSphere(
                atmosphere, rain_volume,
                r, mu);

            const auto transmittance = aten::exp(-extinction * optical_depth);
            AT_ASSERT(transmittance <= 1.0F);

            aten::sky::WriteTexture2D(
                transmittance_in_rain_volume_texture,
                aten::vec3(transmittance),
                x, y);
        }

        __global__ void ComputeAiryFunctionKernel(idaten::SurfaceTexture spectrum_srgb_tex)
        {
            const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
            const int32_t z = blockIdx.z * blockDim.z + threadIdx.z;
            if (x >= aten::rainbow::THETA_WIDTH
                || y >= aten::rainbow::WAVELENGTH_WIDTH
                || z >= aten::rainbow::A_WIDTH)
            {
                return;
            }

            // TODO
            const auto intensity = aten::rainbow::ComputeAiryFunction(x, y, z);
            aten::sky::WriteTexture2D(spectrum_srgb_tex, aten::vec3(intensity), x, y);
        }

        __global__ void FillDropletRadiusInRainVolume(
            idaten::SurfaceTexture droplet_radius_texture,
            uint32_t* random_values,
            const float mu,
            const float sigma)
        {
            const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
            const int32_t z = blockIdx.z * blockDim.z + threadIdx.z;
            if (x >= aten::rainbow::DROPLET_RADIUS_TEX_SIZE
                || y >= aten::rainbow::DROPLET_RADIUS_TEX_SIZE
                || z >= aten::rainbow::DROPLET_RADIUS_TEX_SIZE)
            {
                return;
            }

            const auto id = z * (aten::rainbow::DROPLET_RADIUS_TEX_SIZE * aten::rainbow::DROPLET_RADIUS_TEX_SIZE)
                + y * aten::rainbow::DROPLET_RADIUS_TEX_SIZE
                + x;

            const auto rnd = random_values[id];
            const auto frame = 0;
            const auto scramble = rnd * 0x1fe3434f * (((frame + rnd) + 133 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
            aten::CMJ sampler;
            sampler.init(
                (frame + rnd) % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM),
                0,
                scramble);

            const auto u = sampler.nextSample();
            //const auto droplet_radius = aten::rainbow::ComputeInverseNormalDistributionCDF(u, mu, sigma);
            //aten::sky::WriteTexture3D(droplet_radius_texture, aten::vec3(droplet_radius), x, y, z);
        }
    }

    static void PreComputeRainbowValues(
        const bool need_to_compute_atmosphere_transmittance,
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::aabb& rain_volume,
        aten::rainbow::PreComputeTextureManager<idaten::SurfaceTexture, idaten::SurfaceTexture>& rainbow_textures,
        const float intensity_rainfall_rate)
    {
        dim3 thread_per_block(16, 16);

        // Transmittance を計算.
        dim3 transmittance_block_per_grid(
            (aten::sky::TRANSMITTANCE_TEXTURE_WIDTH + thread_per_block.x - 1) / thread_per_block.x,
            (aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT + thread_per_block.y - 1) / thread_per_block.y);

        if (need_to_compute_atmosphere_transmittance) {
            ComputeTransmittanceToTopAtmosphereBoundaryTexture << <transmittance_block_per_grid, thread_per_block >> > (
                atmosphere,
                rainbow_textures.transmittance_texture);
            checkCudaKernel(ComputeTransmittanceToTopAtmosphereBoundaryTexture);
        }

        const auto extinction = aten::rainbow::ComputeExtinctionInRain(intensity_rainfall_rate);

        ComputeTransmittanceInRainVolumeTexture << <transmittance_block_per_grid, thread_per_block >> > (
            atmosphere,
            rainbow_textures.transmittance_in_rain_volume_texture,
            rain_volume,
            extinction);
        checkCudaKernel(ComputeTransmittanceInRainVolumeTexture);

        // For 3 dimension cuda kernel.
        thread_per_block = dim3(8, 8, 8);

        // Compute Airy function.
        dim3 airy_func_block_per_grid(
            (aten::rainbow::THETA_WIDTH + thread_per_block.x - 1) / thread_per_block.x,
            (aten::rainbow::WAVELENGTH_WIDTH + thread_per_block.y - 1) / thread_per_block.y,
            (aten::rainbow::A_WIDTH + thread_per_block.z - 1) / thread_per_block.z);

        ComputeAiryFunctionKernel << <airy_func_block_per_grid, thread_per_block >> > (
            rainbow_textures.spectrum_srgb_tex);
        checkCudaKernel(ComputeAiryFunctionKernel);

#if 0
        // Fill droplet radius.
        dim3 droplet_radius_block_per_grid(
            (aten::rainbow::DROPLET_RADIUS_TEX_SIZE + thread_per_block.x - 1) / thread_per_block.x,
            (aten::rainbow::DROPLET_RADIUS_TEX_SIZE + thread_per_block.y - 1) / thread_per_block.y,
            (aten::rainbow::DROPLET_RADIUS_TEX_SIZE + thread_per_block.z - 1) / thread_per_block.z);

        constexpr auto mu = 0.5_mm;

        // NOTE:
        // Sigma for 95% of normal distribution is precisely 1.96.
        // We can often see 2.0. But, it's approximation. It's not correct mathematically.
        // In this case, the target range is 0.3 - 0.7F.
        constexpr auto sigma = 0.2_mm / 1.96F;

        FillDropletRadiusInRainVolume << <airy_func_block_per_grid, thread_per_block >> > (
            rainbow_textures.droplet_radius_texture,
            random_values_.data(),
            mu, sigma);
        checkCudaKernel(FillDropletRadiusInRainVolume);
#endif
    }

    void RainbowModel::PreCompute()
    {
        PreComputeRainbowValues(
            true,
            atmosphere_,
            rain_volume_,
            textures_,
            intensity_rainfall_rate);
    }

    namespace {
        __global__ void RenderRainbow(
            cudaSurfaceObject_t dst,
            uint32_t* random_values,
            const int32_t width, const int32_t height,
            const aten::CameraParameter camera,
            const aten::sky::AtmosphereParameters atmosphere,
            const idaten::SurfaceTexture transmittance_texture,
            const idaten::SurfaceTexture transmittance_in_rain_volume_texture,
            const idaten::SurfaceTexture droplet_radius_tex,
            const aten::vec3 sun_direction,
            const aten::vec3 earth_center, // [km]
            const aten::aabb rain_volume,  // [km x km x km]
            const float intensity_rainfall_rate,    // [mm/h]
            const idaten::SurfaceTexture spectrum_srgb_tex,
            const aten::vec3 sun_radiance_to_luminance,
            const aten::vec3 white_point)
        {
            const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width || y >= height) {
                return;
            }

            const auto id = y * width + x;

            const auto rnd = random_values[id];
            const auto frame = 0;
            const auto scramble = rnd * 0x1fe3434f * (((frame + rnd) + 133 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
            aten::CMJ sampler;
            sampler.init(
                (frame + rnd) % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM),
                0,
                scramble);

            const float s = x / static_cast<float>(camera.width);
            const float t = y / static_cast<float>(camera.height);

            // TODO
            // Pinhole?
            AT_NAME::CameraSampleResult camsample;
            AT_NAME::PinholeCamera::sample(&camsample, &camera, s, t);

            const auto camera_pos{ camsample.r.org };
            const auto view_dir{ camsample.r.dir };

            aten::vec3 rainbow_radiance {
                aten::rainbow::AdvanceRainVolumeIntegral(
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
                    spectrum_srgb_tex)
            };

            // The rainbow phase texture is already converted from spectrum to linear sRGB.
            // Do not apply the RGB wavelength-to-luminance factors again.
            //rainbow_radiance *= sun_radiance_to_luminance;
            rainbow_radiance = aten::vmax(rainbow_radiance, 0.0F);

            // TODO
            // Tone mapping.
            // white point (RGB=1.0（白））に対する比率の負値のexponential -> 強い値ほど減衰（ゼロに近い）.
            // それを 1.0 から引くことで、結果強い値が大きくなる.
            // exposure は全体の明るさを調整するための係数.
            aten::vec3 color{
                aten::vec3(1.0F) - aten::exp(
                    -rainbow_radiance / white_point * aten::sky::EXPOSURE * aten::rainbow::RAINBOW_EXPOSURE_SCALE)
            };

            surf2Dwrite(
                make_float4(color.x, color.y, color.z, 1.0F),
                dst,
                x * sizeof(float4), y,
                cudaBoundaryModeTrap);
        }
    }

    void RainbowModel::Render(
        GLuint gltex,
        const int32_t width,
        const int32_t height,
        // const float sun_zenith_angle_radians,
        // const float sun_azimuth_angle_radians,
        const aten::CameraParameter& camera)
    {
        if (!m_glimg.IsValid()) {
            m_glimg.init(gltex, CudaGLRscRegisterType::ReadWrite);
        }

        constexpr auto sun_angle = aten::Deg2Rad(20.0F);

        // TODO
        aten::vec3 sun_direction{
            0.0F,
            aten::sin(sun_angle),
            aten::cos(sun_angle),
        };
        sun_direction = normalize(sun_direction);

        const aten::vec3 earth_center{
            0.0F,
            -aten::sky::BottomRadius.as(aten::MeterUnit::km),
            0.0F,
        };

        dim3 thread_per_block(16, 16);
        dim3 block_per_grid(
            (width + thread_per_block.x - 1) / thread_per_block.x,
            (height + thread_per_block.y - 1) / thread_per_block.y);

        CudaGLResourceMapper<decltype(m_glimg)> rscmap(m_glimg);
        auto output_surface = m_glimg.bind();

        RenderRainbow << <block_per_grid, thread_per_block >> > (
            output_surface,
            random_values_.data(),
            width, height,
            camera,
            atmosphere_,
            textures_.transmittance_texture,
            textures_.transmittance_in_rain_volume_texture,
            textures_.droplet_radius_tex,
            sun_direction,
            earth_center,
            rain_volume_,
            intensity_rainfall_rate,
            textures_.spectrum_srgb_tex,
            sun_radiance_to_luminance_,
            white_point_);
        checkCudaKernel(RenderRainbow);

        m_glimg.unbind();
    }
}

namespace idaten {
    void Atmosphere::InitRainbow()
    {
        idaten::rainbow::InitRandom(random_values_);
    }

    void Atmosphere::PreComputeRainbow()
    {
        idaten::rainbow::PreComputeRainbowValues(
            false,
            sky_model_.atmosphere_,
            rainbow_model_.rain_volume_,
            rainbow_textures_,
            aten::rainbow::RainbowModel::intensity_rainfall_rate);
    }
}
