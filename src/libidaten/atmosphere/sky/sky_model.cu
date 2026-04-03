#include "sky_model_device.h"

#include "atmosphere/sky/sky_common.h"
#include "atmosphere/sky/sky_coord_convert.h"
#include "atmosphere/sky/sky_compute.h"
#include "atmosphere/sky/sky_constants.h"
#include "atmosphere/sky/sky_precompute_textures.h"
#include "atmosphere/sky/sky_render.h"
#include "atmosphere/sky/unit_quantity.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

namespace idaten::sky {
    void SkyModel::Init()
    {
        textures_.Init(pre_compute_textures_host_);
        InitParameters();
    }

    namespace {
        // Transmittance を計算.
        __global__ void ComputeTransmittanceToTopAtmosphereBoundaryTexture(
            const aten::sky::AtmosphereParameters atmosphere,
            aten::sky::PreComputeTextures textures)
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

            aten::sky::WriteTexture2D(textures.transmittance_texture, transmittance, x, y);
        }

        __device__ aten::vec2 IRRADIANCE_TEXTURE_SIZE{
            aten::sky::IRRADIANCE_TEXTURE_WIDTH,
            aten::sky::IRRADIANCE_TEXTURE_HEIGHT,
        };

        // 太陽からの入射放射輝度から指定された点での放射照度を計算する.
        __global__ void ComputeDirectIrradianceTexture(
            const aten::sky::AtmosphereParameters atmosphere,
            aten::sky::PreComputeTextures textures)
        {
            const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= aten::sky::IRRADIANCE_TEXTURE_WIDTH || y >= aten::sky::IRRADIANCE_TEXTURE_HEIGHT) {
                return;
            }

            const aten::vec2 frag_coord{
                static_cast<float>(x) + 0.5F,
                static_cast<float>(y) + 0.5F,
            };

            float r;
            float mu_s;

            aten::sky::GetRMuSFromIrradianceTextureUv(
                atmosphere, frag_coord / IRRADIANCE_TEXTURE_SIZE, r, mu_s);

            aten::vec3 irradiance{
                ComputeDirectIrradiance(atmosphere, textures.transmittance_texture, r, mu_s)
            };

            aten::sky::WriteTexture2D(textures.delta_irradiance_texture, irradiance, x, y);
            aten::sky::WriteTexture2D(textures.irradiance_texture, aten::vec3(0.0F), x, y);
        }

        // 論文内の 4. Precomputations の Angular precision の部分で説明されている、太陽光からの単一散乱のテクスチャを計算するためのシェーダ.
        //  delta_rayleigh: レイリー散乱の係数のみ. luminanceとの乗算はしていない.
        //  delta_mie: ミー散乱の係数のみ. luminanceとの乗算はしていない.
        //  scattering: 3Dテクスチャ. vec4.rgb に C*（レイリー散乱）, vec4.a に CM.r（ミー散乱） の値が入るテクスチャ. layer 回レンダリングすることで、3次元に値を格納する.
        //  single_mie_scattering: CM （ミー散乱）のみを格納するテクスチャ.
        __global__ void ComputeSingleScatteringTexture(
            const aten::sky::AtmosphereParameters atmosphere,
            const aten::mat4 luminance_from_radiance,
            const int32_t layer,
            aten::sky::PreComputeTextures textures)
        {
            const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= aten::sky::SCATTERING_TEXTURE_WIDTH || y >= aten::sky::SCATTERING_TEXTURE_HEIGHT) {
                return;
            }

            float r;
            float mu;
            float mu_s;
            float nu;
            bool ray_r_mu_intersects_ground;

            const aten::vec3 frag_coord{
                static_cast<float>(x) + 0.5F,
                static_cast<float>(y) + 0.5F,
                static_cast<float>(layer) + 0.5F,
            };

            aten::sky::GetRMuMuSNuFromScatteringTextureFragCoord(
                atmosphere,
                frag_coord,
                r, mu, mu_s, nu, ray_r_mu_intersects_ground);

            aten::vec3 rayleigh;
            aten::vec3 mie;

            aten::sky::single_scattering::ComputeSingleScattering(
                atmosphere,
                textures.transmittance_texture,
                r, mu, mu_s, nu, ray_r_mu_intersects_ground,
                rayleigh, mie);

            aten::sky::WriteTexture3D(textures.delta_rayleigh_scattering_texture, rayleigh, x, y, layer);
            aten::sky::WriteTexture3D(textures.delta_mie_scattering_texture, mie, x, y, layer);

            rayleigh = luminance_from_radiance * rayleigh;
            mie = luminance_from_radiance * mie;

            aten::sky::WriteTexture3D(
                textures.scattering_texture,
                aten::vec4(rayleigh.r, rayleigh.g, rayleigh.b, mie.r),
                x, y, layer);
            aten::sky::WriteTexture3D(textures.optional_single_mie_scattering_texture, mie, x, y, layer);
        }

        // ΔJ を計算する.
        __global__ void ComputeScatteringDensityTexture(
            const aten::sky::AtmosphereParameters atmosphere,
            const aten::mat4 luminance_from_radiance,
            const int32_t layer,
            aten::sky::PreComputeTextures textures,
            const int32_t scattering_order)
        {
            const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= aten::sky::SCATTERING_TEXTURE_WIDTH || y >= aten::sky::SCATTERING_TEXTURE_HEIGHT) {
                return;
            }

            const aten::vec3 frag_coord{
                static_cast<float>(x) + 0.5F,
                static_cast<float>(y) + 0.5F,
                static_cast<float>(layer) + 0.5F,
            };

            float r;
            float mu;
            float mu_s;
            float nu;
            bool ray_r_mu_intersects_ground;

            aten::sky::GetRMuMuSNuFromScatteringTextureFragCoord(
                atmosphere,
                frag_coord,
                r, mu, mu_s, nu, ray_r_mu_intersects_ground);

            const auto scattering_density{
                aten::sky::ComputeScatteringDensity(
                    atmosphere,
                    textures.transmittance_texture,
                    textures.delta_rayleigh_scattering_texture,
                    textures.delta_mie_scattering_texture,
                    textures.delta_multiple_scattering_texture,
                    textures.delta_irradiance_texture,
                    r, mu, mu_s, nu,
                    scattering_order)
            };

            aten::sky::WriteTexture3D(
                textures.delta_scattering_density_texture,
                scattering_density,
                x, y, layer);
        }

        // E = E + ΔE を計算するための、太陽光でない入射する放射照度 ΔE を計算する.
        __global__ void ComputeIndirectIrradianceTexture(
            const aten::sky::AtmosphereParameters atmosphere,
            const aten::mat4 luminance_from_radiance,
            aten::sky::PreComputeTextures textures,
            const int32_t scattering_order)
        {
            const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= aten::sky::IRRADIANCE_TEXTURE_WIDTH || y >= aten::sky::IRRADIANCE_TEXTURE_HEIGHT) {
                return;
            }

            const aten::vec2 frag_coord{
                static_cast<float>(x) + 0.5F,
                static_cast<float>(y) + 0.5F,
            };

            float r;
            float mu_s;
            aten::sky::GetRMuSFromIrradianceTextureUv(
                atmosphere,
                frag_coord / IRRADIANCE_TEXTURE_SIZE,
                r, mu_s);

            auto delta_irradiance{
                ComputeIndirectIrradiance(atmosphere,
                    textures.delta_rayleigh_scattering_texture,
                    textures.delta_mie_scattering_texture,
                    textures.delta_multiple_scattering_texture,
                    r, mu_s,
                    scattering_order)
            };

            const aten::vec2 uv {
                (x + 0.5F) / IRRADIANCE_TEXTURE_SIZE.x,
                (y + 0.5F) / IRRADIANCE_TEXTURE_SIZE.y,
            };

            const auto curr_irradiance{
                aten::sky::SampleTexture2D(textures.irradiance_texture, uv) };
            delta_irradiance = luminance_from_radiance * delta_irradiance;

            aten::sky::WriteTexture2D(
                textures.irradiance_texture,
                curr_irradiance + delta_irradiance,
                x, y);
        }

        // S = S + ΔS を計算するための ΔS を計算する.
        __global__ void ComputeMultipleScatteringTexture(
            const aten::sky::AtmosphereParameters atmosphere,
            const aten::mat4 luminance_from_radiance,
            const int32_t layer,
            aten::sky::PreComputeTextures textures)
        {
            const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= aten::sky::SCATTERING_TEXTURE_WIDTH || y >= aten::sky::SCATTERING_TEXTURE_HEIGHT) {
                return;
            }

            const aten::vec3 frag_coord{
                static_cast<float>(x) + 0.5F,
                static_cast<float>(y) + 0.5F,
                static_cast<float>(layer) + 0.5F,
            };

            float r;
            float mu;
            float mu_s;
            float nu;
            bool ray_r_mu_intersects_ground;

            aten::sky::GetRMuMuSNuFromScatteringTextureFragCoord(
                atmosphere,
                frag_coord,
                r, mu, mu_s, nu, ray_r_mu_intersects_ground);

            const auto delta_multiple_scattering{
                ComputeMultipleScattering(
                    atmosphere,
                    textures.transmittance_texture,
                    textures.delta_scattering_density_texture,
                    r, mu, mu_s, nu,
                    ray_r_mu_intersects_ground)
            };

            const aten::vec3 uvw{
                (x + 0.5F) / aten::sky::SCATTERING_TEXTURE_WIDTH,
                (y + 0.5F) / aten::sky::SCATTERING_TEXTURE_HEIGHT,
                (layer + 0.5F) / aten::sky::SCATTERING_TEXTURE_DEPTH,
            };

            const auto curr_scattering{
                aten::sky::SampleTexture3D(textures.scattering_texture, uvw) };

            aten::sky::WriteTexture3D(
                textures.delta_multiple_scattering_texture,
                delta_multiple_scattering,
                x, y, layer);

            const auto phase = aten::sky::RayleighPhaseFunction(nu);

            // C∗ = S_R[L0] + S[L∗]/P_R として保存.
            // P_R は後で乗算して、P_R・S_R[L0]+S[L∗] として計算するため.
            aten::sky::WriteTexture3D(
                textures.scattering_texture,
                curr_scattering + delta_multiple_scattering / phase,
                x, y, layer);
        }
    }

    void SkyModel::PreCompute()
    {
        dim3 thread_per_block(16, 16);
        dim3 transmittance_block_per_grid(
            (aten::sky::TRANSMITTANCE_TEXTURE_WIDTH + thread_per_block.x - 1) / thread_per_block.x,
            (aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT + thread_per_block.y - 1) / thread_per_block.y);
        dim3 irradiance_block_per_grid(
            (aten::sky::IRRADIANCE_TEXTURE_WIDTH + thread_per_block.x - 1) / thread_per_block.x,
            (aten::sky::IRRADIANCE_TEXTURE_HEIGHT + thread_per_block.y - 1) / thread_per_block.y);
        dim3 scattering_block_per_grid(
            (aten::sky::SCATTERING_TEXTURE_WIDTH + thread_per_block.x - 1) / thread_per_block.x,
            (aten::sky::SCATTERING_TEXTURE_HEIGHT + thread_per_block.y - 1) / thread_per_block.y);

        // Transmittance を計算.
        ComputeTransmittanceToTopAtmosphereBoundaryTexture << <transmittance_block_per_grid, thread_per_block >> > (
            atmosphere_,
            textures_);
        checkCudaKernel(ComputeTransmittanceToTopAtmosphereBoundaryTexture);

        // 最初の ΔE を計算.
        // 太陽からの入射放射輝度から指定された点での放射照度を計算する.
        ComputeDirectIrradianceTexture << <irradiance_block_per_grid, thread_per_block >> > (
            atmosphere_,
            textures_);
        checkCudaKernel(ComputeDirectIrradianceTexture);

        // 最初の ΔS を計算.
        // 太陽光（一方向）からの単一散乱.
        for (int32_t z = 0; z < aten::sky::SCATTERING_TEXTURE_DEPTH; z++) {
            ComputeSingleScatteringTexture << <scattering_block_per_grid, thread_per_block >> > (
                atmosphere_,
                luminance_from_radiance_,
                z,
                textures_);
            checkCudaKernel(ComputeSingleScatteringTexture);
        }

        // ここまでで、1st scattering order は計算済み。次に、2nd scattering order 以降を順番に計算していく.
        for (int32_t scattering_order = 2; scattering_order <= NUM_SCATTERING; scattering_order++)
        {
            // ΔJ を計算.
            for (int32_t z = 0; z < aten::sky::SCATTERING_TEXTURE_DEPTH; z++) {
                ComputeScatteringDensityTexture << <scattering_block_per_grid, thread_per_block >> > (
                    atmosphere_,
                    luminance_from_radiance_,
                    z,
                    textures_,
                    scattering_order);
                checkCudaKernel(ComputeScatteringDensityTexture);
            }

            // ΔE を計算して、E = E + ΔE する.
            ComputeIndirectIrradianceTexture << <irradiance_block_per_grid, thread_per_block >> > (
                atmosphere_,
                luminance_from_radiance_,
                textures_,
                scattering_order - 1);
            checkCudaKernel(ComputeIndirectIrradianceTexture);

            // ΔS を計算して、S = S + ΔS する.
            for (int32_t z = 0; z < aten::sky::SCATTERING_TEXTURE_DEPTH; z++) {
                ComputeMultipleScatteringTexture << <scattering_block_per_grid, thread_per_block >> > (
                    atmosphere_,
                    luminance_from_radiance_,
                    z,
                    textures_);
                checkCudaKernel(ComputeMultipleScatteringTexture);
            }
        }
    }

    namespace {
        __global__ void RenderSkyKernel(
            cudaSurfaceObject_t dst,
            int32_t width, int32_t height,
            const aten::CameraParameter camera,
            const aten::sky::AtmosphereParameters atmosphere,
            const aten::sky::PreComputeTextures textures,
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

            auto sky_luminance{
                aten::sky::RenderSky(
                    x, y,
                    camera,
                    atmosphere, textures,
                    sun_radiance_to_luminance, sky_radiance_to_luminance,
                    sun_direction,
                    earth_center,
                    sun_size)
            };

            // TODO
            // Tone mapping.
            // white point (RGB=1.0（白））に対する比率の負値のexponential -> 強い値ほど減衰（ゼロに近い）.
            // それを 1.0 から引くことで、結果強い値が大きくなる.
            // exposure は全体の明るさを調整するための係数.
            aten::vec3 color{
                aten::vec3(1.0F) - aten::exp(-sky_luminance / white_point * aten::sky::EXPOSURE)
            };

            surf2Dwrite(
                make_float4(color.x, color.y, color.z, 1.0F),
                dst,
                x * sizeof(float4), y,
                cudaBoundaryModeTrap);
        }
    }

    // NOTE
    // camera parameters has to be specified based on km unit.
    void SkyModel::Render(
        GLuint gltex,
        const int32_t width,
        const int32_t height,
        const aten::CameraParameter& camera)
    {
        if (!m_glimg.IsValid()) {
            m_glimg.init(gltex, CudaGLRscRegisterType::ReadWrite);
        }

        // TODO
        const auto sun_zenith_angle_radians = 1.3F;
        const auto sun_azimuth_angle_radians = 2.9F;

        const aten::vec3 sun_direction{
            aten::sin(sun_zenith_angle_radians) * aten::cos(sun_azimuth_angle_radians),
            aten::cos(sun_zenith_angle_radians),
            aten::sin(sun_zenith_angle_radians) * aten::sin(sun_azimuth_angle_radians)
        };

        const aten::vec3 earth_center{
            0.0F,
            -aten::sky::BottomRadius.as(aten::MeterUnit::km),
            0.0F,
        };

        const auto sun_size = aten::cos(aten::sky::SunAngularRadius);

        dim3 thread_per_block(16, 16);
        dim3 block_per_grid(
            (width + thread_per_block.x - 1) / thread_per_block.x,
            (height + thread_per_block.y - 1) / thread_per_block.y);

        CudaGLResourceMapper<decltype(m_glimg)> rscmap(m_glimg);
        auto output_surface = m_glimg.bind();

        RenderSkyKernel << <block_per_grid, thread_per_block >> > (
            output_surface,
            width, height,
            camera,
            atmosphere_, textures_,
            sun_radiance_to_luminance_, sky_radiance_to_luminance_,
            sun_direction,
            earth_center,
            sun_size,
            white_point_);
        checkCudaKernel(RenderSkyKernel);

        m_glimg.unbind();
    }
}
