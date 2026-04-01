#include "atmosphere/sky/sky_model.h"

#include "atmosphere/sky/sky_common.h"
#include "atmosphere/sky/sky_compute.h"
#include "atmosphere/sky/sky_coord_convert.h"
#include "atmosphere/sky/sky_render.h"

#include "atmosphere/sky/unit_quantity.h"

namespace aten::sky {
    namespace {
        void SetDensityProfileLayer(
            aten::sky::DensityProfileLayer& dst,
            const aten::sky::DensityProfileLayer& src)
        {
            // m -> km
            dst.width = aten::Length::as(src.width, MeterUnit::km);

            dst.exp_term = src.exp_term;

            // m^-1 -> km^-1
            dst.inv_height_scale = aten::InverseLength::as(src.inv_height_scale, MeterUnit::km);
            dst.linear_term = aten::InverseLength::as(src.linear_term, MeterUnit::km);

            dst.constant_term = src.constant_term;
        }
    }

    void SkyModel::Init()
    {
        textures_.Init();

        using Irrandiance = InverseLength;

        std::vector<float> wavelengths;
        std::vector<float> solar_irradiance;
        std::vector<float> rayleigh_scattering;
        std::vector<float> mie_scattering;
        std::vector<float> mie_extinction;
        std::vector<float> absorption_extinction;
        std::vector<float> ground_albedo;

        for (int32_t l = LambdaMin; l <= LambdaMax; l += 10)
        {
            // 波長の範囲が
            //  LambdaMin = 360[mm]
            //  LambdaMax = 830[mm]
            // で定義されていて、これを mm (1e-3) -> micro meter (1e-6) に変換.
            const auto lambda = static_cast<float>(l) * 1e-3F;

            wavelengths.emplace_back(l);

            solar_irradiance.emplace_back(ConstantSolarIrradiance);

            // kRayleighは、式(1)での exp(-h/H) の前にある項の部分のλの依存性がない部分で定数.
            // ここでは、式(1)のλの依存性の部分を計算して、rayleigh_scatteringに格納する.
            /*
                https://gemini.google.com/share/b1f3242d1df2
              標準状態（STP: 0°C, 1013.25 hPa）における大気の値を採用:
                空気の屈折率 (n): 約 1.000293
                分子数密度 (N): ロシュミット数（Loschmidt constant）を用いる.
	                N = 2.545 × 10^25 [molecules/m^3]
                円周率 (\pi): 3.14159265


              β_R(λ) = 8pi^3(n^2-1)^2 / 3Nλ^4 を、変数であるλ 以外の部分（係数 k）としてまとめる.
              k  = 8pi^3(n^2-1)^2 / 3N

              k_SI = 1.1156 × 10^-30

              この k_SI は、波長 λ を メートル (m) 単位で入れた時の値.

              コードでは λ に μm 単位の数値を入れます。1[μm] = 10^-6 [m] なので、
              分母にある λ^4 は次のように置き換わります。
                λ^4[m] = (λ × 10^-6 [μm])^4 = λ[m]^4 × 10^-24

              これを元の式に代入すると、定数部分は 10^-24 で割られる（つまり 10^24 倍される）ことになる.
                k_μm = k_SI × 10^24 = 1.1156 × 10^-30 × 10^24 = 1.1156 × 10^-6
            */
            rayleigh_scattering.emplace_back(Rayleigh * aten::pow(lambda, -4.0F));

            /*
                https://gemini.google.com/share/269fb9c1472b
              オングストローム（Angstrom）の濁度公式（または混濁係数の式）は、
              大気中のエアロゾルによる光の散乱・吸収による減衰を表すための経験式です。
              この公式は、波長 λ におけるエアロゾルの光学的厚さ（オプティカル・デプス）
              τ_α(λ) を次のように定義します。

              τ_α(λ) = βλ^{-α}

              各変数の意味

              τ_α(λ) : 波長 λ におけるエアロゾルの光学的厚さ
                - 光学的厚さは無次元

              λ : 波長（通常はマイクロメートル 単位で表されます）。

              β（オングストローム濁度係数）:
                - 大気中のエーロゾルの総量を反映する係数です。
                - 値が大きいほど、大気が濁っている（粒子数が多い）ことを示します。
                - λ = 1[μm] の時の光学的厚さ

              α（オングストローム指数）:
                - 粒子のサイズ分布に関連する指数です。
                - 一般的に、粒子が小さいほど α は大きくなり（例：煙やもやでは 1.3 - 2.0 程度）
                  粒子が大きいほど α は小さくなります（例：砂塵や雲では 0 に近づく）
            */
            /*
                https://gemini.google.com/share/b1f3242d1df2
              単位（次元）で分解すると:
                MieAngstromBeta: 5.328e-3（無次元）
                MieScaleHeight: 1200.0[m] ：ミー散乱の分布高度（スケールハイト）
                lambda: マイクロメートル単位の数値
                pow(lambda, -0.0): lambda^0 = 1（無次元）

              これらを組み合わせると：
                単位 = 無次元 / m × 無次元 = m^-1
            */
            const auto mie = MieAngstromBeta / MieScaleHeight * aten::pow(lambda, -MieAngstromAlpha);

            // https://gemini.google.com/share/269fb9c1472b
            mie_scattering.emplace_back(mie * MieSingleScatteringAlbedo);
            mie_extinction.emplace_back(mie);

            // TODO
            // Support Ozon.
            absorption_extinction.emplace_back(0.0F);

            // 理想的には波長ごとに地面での反射率は変わるべき?
            ground_albedo.emplace_back(GroundAlbedo);
        }

        const DensityProfileLayer rayleigh_layer{
            0.0F,
            1.0F,
            -1.0F / RayleighScaleHeight,
            0.0F,
            0.0F
        };

        const DensityProfileLayer mie_layer{
            0.0F,
            1.0F,
            -1.0F / MieScaleHeight,
            0.0F,
            0.0F
        };

        const vec3 rgb_lambdas{
            LambdaR,
            LambdaG,
            LambdaB,
        };

        atmosphere_.solar_irradiance = InterpolateFactorByRGBLambda(solar_irradiance, wavelengths, rgb_lambdas);

        atmosphere_.sun_angular_radius = SunAngularRadius;

        // m -> km
        atmosphere_.bottom_radius = aten::Length::as(BottomRadius, MeterUnit::km);
        atmosphere_.top_radius = aten::Length::as(TopRadius, MeterUnit::km);

        // NOTE:
        // altitude < layers[0].width ? layers[0] : layers[1] で分岐される.
        // layers[0].width = 0.0F なので、常に layers[1] が利用される.
        SetDensityProfileLayer(atmosphere_.rayleigh_density.layers[1], rayleigh_layer);

        // m^-1 -> km^-1
        atmosphere_.rayleigh_scattering = InterpolateFactorByRGBLambda(rayleigh_scattering, wavelengths, rgb_lambdas);
        atmosphere_.rayleigh_scattering.x = aten::InverseLength::as(atmosphere_.rayleigh_scattering.x, MeterUnit::km);
        atmosphere_.rayleigh_scattering.y = aten::InverseLength::as(atmosphere_.rayleigh_scattering.y, MeterUnit::km);
        atmosphere_.rayleigh_scattering.z = aten::InverseLength::as(atmosphere_.rayleigh_scattering.z, MeterUnit::km);

        // NOTE:
        // altitude < layers[0].width ? layers[0] : layers[1] で分岐される.
        // layers[0].width = 0.0F なので、常に layers[1] が利用される.
        SetDensityProfileLayer(atmosphere_.mie_density.layers[1], mie_layer);

        atmosphere_.mie_scattering = InterpolateFactorByRGBLambda(mie_scattering, wavelengths, rgb_lambdas);

        // m^-1 -> km^-1
        atmosphere_.mie_scattering.x = aten::InverseLength::as(atmosphere_.mie_scattering.x, MeterUnit::km);
        atmosphere_.mie_scattering.y = aten::InverseLength::as(atmosphere_.mie_scattering.y, MeterUnit::km);
        atmosphere_.mie_scattering.z = aten::InverseLength::as(atmosphere_.mie_scattering.z, MeterUnit::km);

        // m^-1 -> km^-1
        atmosphere_.mie_extinction = InterpolateFactorByRGBLambda(mie_extinction, wavelengths, rgb_lambdas);
        atmosphere_.mie_extinction.x = aten::InverseLength::as(atmosphere_.mie_extinction.x, MeterUnit::km);
        atmosphere_.mie_extinction.y = aten::InverseLength::as(atmosphere_.mie_extinction.y, MeterUnit::km);
        atmosphere_.mie_extinction.z = aten::InverseLength::as(atmosphere_.mie_extinction.z, MeterUnit::km);

        atmosphere_.mie_phase_function_g = MiePhaseFunctionG;

        // TODO
        // Ozon

        atmosphere_.ground_albedo = InterpolateFactorByRGBLambda(ground_albedo, wavelengths, rgb_lambdas);

        atmosphere_.mu_s_min = aten::cos(MaxSunZenithAngle);

        sun_radiance_to_luminance_ = aten::sky::ComputeSpectralRadianceToLuminanceFactors(wavelengths, solar_irradiance, 0);
        sky_radiance_to_luminance_ = aten::sky::ComputeSpectralRadianceToLuminanceFactors(wavelengths, solar_irradiance, -3);

        // TODO
        // For tone mapping.
        white_point_ = ConvertSpectrumToLinearSrgb(wavelengths, solar_irradiance);
        const auto white_point_avg = (white_point_.r + white_point_.g + white_point_.b) / 3.0F;
        white_point_ /= white_point_avg;
    }

    namespace {
        // Transmittance を計算.
        void ComputeTransmittanceToTopAtmosphereBoundaryTexture(
            const aten::sky::AtmosphereParameters& atmosphere,
            const int32_t x, const int32_t y,
            aten::sky::PreComputeTextures& textures)
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

            WriteTexture2D(textures.transmittance_texture, transmittance, x, y);
        }

        static const aten::vec2 IRRADIANCE_TEXTURE_SIZE{
            aten::sky::IRRADIANCE_TEXTURE_WIDTH,
            aten::sky::IRRADIANCE_TEXTURE_HEIGHT,
        };

        // 太陽からの入射放射輝度から指定された点での放射照度を計算する.
        void ComputeDirectIrradianceTexture(
            const aten::sky::AtmosphereParameters& atmosphere,
            const int32_t x, const int32_t y,
            aten::sky::PreComputeTextures& textures)
        {
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

            WriteTexture2D(textures.delta_irradiance_texture, irradiance, x, y);
            WriteTexture2D(textures.irradiance_texture, aten::vec3(0.0F), x, y);
        }

        // 論文内の 4. Precomputations の Angular precision の部分で説明されている、太陽光からの単一散乱のテクスチャを計算するためのシェーダ.
        //  delta_rayleigh: レイリー散乱の係数のみ. luminanceとの乗算はしていない.
        //  delta_mie: ミー散乱の係数のみ. luminanceとの乗算はしていない.
        //  scattering: 3Dテクスチャ. vec4.rgb に C*（レイリー散乱）, vec4.a に CM.r（ミー散乱） の値が入るテクスチャ. layer 回レンダリングすることで、3次元に値を格納する.
        //  single_mie_scattering: CM （ミー散乱）のみを格納するテクスチャ.
        void ComputeSingleScatteringTexture(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::mat4& luminance_from_radiance,
            const int32_t x, const int32_t y, const int32_t layer,
            aten::sky::PreComputeTextures& textures)
        {
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

            WriteTexture3D(textures.delta_rayleigh_scattering_texture, rayleigh, x, y, layer);
            WriteTexture3D(textures.delta_mie_scattering_texture, mie, x, y, layer);

            rayleigh = luminance_from_radiance * rayleigh;
            mie = luminance_from_radiance * mie;

            WriteTexture3D(
                textures.scattering_texture,
                aten::vec4(rayleigh.r, rayleigh.g, rayleigh.b, mie.r),
                x, y, layer);
            WriteTexture3D(textures.optional_single_mie_scattering_texture, mie, x, y, layer);
        }

        // ΔJ を計算する.
        void ComputeScatteringDensityTexture(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::mat4& luminance_from_radiance,
            const int32_t x, const int32_t y, const int32_t layer,
            aten::sky::PreComputeTextures& textures,
            const int32_t scattering_order)
        {
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

            WriteTexture3D(
                textures.delta_scattering_density_texture,
                scattering_density,
                x, y, layer);
        }

        // E = E + ΔE を計算するための、太陽光でない入射する放射照度 ΔE を計算する.
        void ComputeIndirectIrradianceTexture(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::mat4& luminance_from_radiance,
            const int32_t x, const int32_t y,
            aten::sky::PreComputeTextures& textures,
            const int32_t scattering_order)
        {
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

            const auto curr_irradiance{ textures.irradiance_texture.AtByXY(x, y) };
            delta_irradiance = luminance_from_radiance * delta_irradiance;

            WriteTexture2D(
                textures.irradiance_texture,
                curr_irradiance + delta_irradiance,
                x, y);
        }

        // S = S + ΔS を計算するための ΔS を計算する.
        void ComputeMultipleScatteringTexture(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::mat4& luminance_from_radiance,
            const int32_t x, const int32_t y, const int32_t layer,
            aten::sky::PreComputeTextures& textures)
        {
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

            const auto curr_scattering{ textures.scattering_texture.AtByXYZ(x, y, layer) };

            WriteTexture3D(
                textures.delta_multiple_scattering_texture,
                delta_multiple_scattering,
                x, y, layer);

            const auto phase = aten::sky::RayleighPhaseFunction(nu);

            // C∗ = S_R[L0] + S[L∗]/P_R として保存.
            // P_R は後で乗算して、P_R・S_R[L0]+S[L∗] として計算するため.
            WriteTexture3D(
                textures.scattering_texture,
                curr_scattering + delta_multiple_scattering / phase,
                x, y, layer);
        }
    }

    void SkyModel::PreCompute()
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
                        textures_);
                }
            }

            // 最初の ΔE を計算.
            // 太陽からの入射放射輝度から指定された点での放射照度を計算する.
#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
            for (int32_t y = 0; y < aten::sky::IRRADIANCE_TEXTURE_HEIGHT; y++) {
                for (int32_t x = 0; x < aten::sky::IRRADIANCE_TEXTURE_WIDTH; x++) {
                    ComputeDirectIrradianceTexture(
                        atmosphere_,
                        x, y,
                        textures_);
                }
            }

            // 最初の ΔS を計算.
            // 太陽光（一方向）からの単一散乱.
#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
            for (int32_t y = 0; y < aten::sky::SCATTERING_TEXTURE_HEIGHT; y++) {
                for (int32_t x = 0; x < aten::sky::SCATTERING_TEXTURE_WIDTH; x++) {
                    for (int32_t z = 0; z < aten::sky::SCATTERING_TEXTURE_DEPTH; z++) {
                        ComputeSingleScatteringTexture(
                            atmosphere_,
                            luminance_from_radiance_,
                            x, y, z,
                            textures_);
                    }
                }
            }

            // ここまでで、1st scattering order は計算済み。次に、2nd scattering order 以降を順番に計算していく.
            for (int32_t scattering_order = 2; scattering_order <= NUM_SCATTERING; scattering_order++)
            {
                // ΔJ を計算.
#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
                for (int32_t y = 0; y < aten::sky::SCATTERING_TEXTURE_HEIGHT; y++) {
                    for (int32_t x = 0; x < aten::sky::SCATTERING_TEXTURE_WIDTH; x++) {
                        for (int32_t z = 0; z < aten::sky::SCATTERING_TEXTURE_DEPTH; z++) {
                            ComputeScatteringDensityTexture(
                                atmosphere_,
                                luminance_from_radiance_,
                                x, y, z,
                                textures_,
                                scattering_order);
                        }
                    }
                }

                // ΔE を計算して、E = E + ΔE する.
#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
                for (int32_t y = 0; y < aten::sky::IRRADIANCE_TEXTURE_HEIGHT; y++) {
                    for (int32_t x = 0; x < aten::sky::IRRADIANCE_TEXTURE_WIDTH; x++) {
                        ComputeIndirectIrradianceTexture(
                            atmosphere_,
                            luminance_from_radiance_,
                            x, y,
                            textures_,
                            scattering_order - 1);
                    }
                }

                // ΔS を計算して、S = S + ΔS する.
#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
                for (int32_t y = 0; y < aten::sky::SCATTERING_TEXTURE_HEIGHT; y++) {
                    for (int32_t x = 0; x < aten::sky::SCATTERING_TEXTURE_WIDTH; x++) {
                        for (int32_t z = 0; z < aten::sky::SCATTERING_TEXTURE_DEPTH; z++) {
                            ComputeMultipleScatteringTexture(
                                atmosphere_,
                                luminance_from_radiance_,
                                x, y, z,
                                textures_);
                        }
                    }
                }
            }
        }
    }

//#pragma optimize("", off)

    // NOTE
    // camera parameters has to be specified based on km unit.
    void SkyModel::Render(
        const int32_t width,
        const int32_t height,
        const aten::CameraParameter& camera,
        Film& dst)
    {
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
            -BottomRadius.as(MeterUnit::km),
            0.0F,
        };

        const auto sun_size = aten::cos(SunAngularRadius);

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
                    auto sky_luminance{
                        RenderSky(
                            x, y,
                            camera,
                            atmosphere_, textures_,
                            sun_radiance_to_luminance_, sky_radiance_to_luminance_,
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
                        aten::vec3(1.0F) - aten::exp(-sky_luminance / white_point_ * EXPOSURE)
                    };

                    if (x == 0) {
                        AT_PRINTF("[%d, %d] : %f, %f, %f\n", x, y, color.r, color.g, color.b);
                    }
                    //AT_PRINTF("[%d, %d] : %f, %f, %f\n", x, y, color.r, color.g, color.b);

                    dst.put(x, y, color);
                }
            }
        }
    }
}
