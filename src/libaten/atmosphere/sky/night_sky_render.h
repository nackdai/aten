#pragma once

#include "atmosphere/sky/sky_render.h"

namespace aten::sky {
    // Phase term used in the paper's moon irradiance equation:
    //   Em(phi, d) = 2 C rm^2 / (3 d^2) * { Eem + Esm * phase(phi) }
    inline AT_HOST_DEVICE_API float ComputeMoonPhaseIrradianceFactor(const float phase_angle)
    {
        constexpr float eps = 1.0e-4F;
        const auto phi = aten::clamp(phase_angle, eps, AT_MATH_PI - eps);
        return 1.0F - aten::sin(phi * 0.5F) * aten::tan(phi * 0.5F) * aten::log(1.0F / aten::tan(phi * 0.25F));
    }

    // Paper Eq. (1): Earthshine irradiance at the Moon, Eem.
    inline AT_HOST_DEVICE_API float ComputeEarthshineIrradianceAtMoon(const float phase_angle)
    {
        constexpr float eps = 1.0e-4F;
        const auto earth_phase = aten::clamp(AT_MATH_PI - phase_angle, eps, AT_MATH_PI - eps);
        const auto phase_factor =
            1.0F - aten::sin(earth_phase * 0.5F) * aten::tan(earth_phase * 0.5F) * aten::log(1.0F / aten::tan(earth_phase * 0.25F));
        return FullEarthshineIrradianceAtMoon * 0.5F * phase_factor;
    }

    // Paper Eq. (2): Moon irradiance at the observer, Em(phi, d).
    inline AT_HOST_DEVICE_API float ComputeMoonIrradiance(
        const float phase_angle,
        const float moon_distance)
    {
        const auto moon_phase_factor = ComputeMoonPhaseIrradianceFactor(phase_angle);
        const auto earthshine_irradiance = ComputeEarthshineIrradianceAtMoon(phase_angle);
        const auto moon_radius = MoonRadius.as(MeterUnit::km);

        return 2.0F * MoonMeanAlbedo * moon_radius * moon_radius / (3.0F * moon_distance * moon_distance)
            * (earthshine_irradiance + SolarIrradianceAtMoon * moon_phase_factor);
    }

    inline AT_HOST_DEVICE_API aten::vec3 ComputeMoonIrradiance(
        const aten::vec3& sun_irradiance,
        const aten::vec3& sun_direction,
        const aten::vec3& moon_direction,
        const float moon_distance)
    {
        const auto cos_phase_angle = aten::clamp(dot(-moon_direction, sun_direction), -1.0F, 1.0F);
        const auto phase_angle = aten::acos(cos_phase_angle);
        const auto em = ComputeMoonIrradiance(phase_angle, moon_distance);

        return sun_irradiance * (em / SolarIrradianceAtMoon);
    }

    inline AT_HOST_DEVICE_API float ComputeMoonAngularRadius(const float moon_distance)
    {
        return aten::asin(aten::clamp(MoonRadius.as(MeterUnit::km) / moon_distance, -1.0F, 1.0F));
    }

    inline AT_DEVICE_API bool ViewRayIntersectsMoonDisk(
        const aten::vec3& view_direction,
        const aten::vec3& moon_direction,
        const float moon_distance)
    {
        // TODO: This is a generic ray-sphere intersection test with the camera
        // at the ray origin. Consider extracting it as a shared helper, e.g.
        // RayIntersectsSphereFromOrigin(ray_direction, sphere_center, radius),
        // and keep this function as the Moon-specific wrapper.
        const auto moon_radius = MoonRadius.as(MeterUnit::km);
        const auto view_center_cos = dot(view_direction, moon_direction);
        const auto projected_distance = view_center_cos * moon_distance;
        const auto discriminant =
            projected_distance * projected_distance
            - (moon_distance * moon_distance - moon_radius * moon_radius);

        return projected_distance > 0.0F && discriminant > 0.0F;
    }

    inline AT_DEVICE_API bool PixelIntersectsMoonDisk(
        const int32_t x,
        const int32_t y,
        const aten::CameraParameter& camera,
        const aten::vec3& moon_direction,
        const float moon_distance)
    {
        const float s = x / static_cast<float>(camera.width);
        const float t = y / static_cast<float>(camera.height);

        AT_NAME::CameraSampleResult camsample;
        AT_NAME::PinholeCamera::sample(&camsample, &camera, s, t);

        return ViewRayIntersectsMoonDisk(camsample.r.dir, moon_direction, moon_distance);
    }

    inline AT_HOST_DEVICE_API float ComputeMoonBackscatter(
        const float phase_angle,
        const float density)
    {
        const auto cos_phase_angle = aten::cos(phase_angle);
        const auto denom = aten::max(
            1.0e-4F,
            1.0F - 2.0F * density * cos_phase_angle + density * density);

        return (1.0F - density * density) / aten::pow(denom, 1.5F);
    }

    // Paper Eq. (3): Hapke-Lommel-Seeliger moon BRDF, using mean albedo.
    inline AT_HOST_DEVICE_API float ComputeMoonBrdf(
        const float cos_theta_i,
        const float cos_theta_r,
        const float phase_angle)
    {
        if (cos_theta_i <= 0.0F || cos_theta_r <= 0.0F) {
            return 0.0F;
        }

        const auto backscatter = ComputeMoonBackscatter(phase_angle, MoonHapkeDensity);
        const auto scattering = ComputeMoonPhaseIrradianceFactor(phase_angle);
        const auto lommel_seeliger = 1.0F / aten::max(cos_theta_i + cos_theta_r, 1.0e-4F);

        return MoonMeanAlbedo * 2.0F / (3.0F * AT_MATH_PI)
            * backscatter
            * scattering
            * lommel_seeliger;
    }

    inline AT_DEVICE_API aten::vec3 GetMoonDiskRadiance(
        const aten::vec3& view_direction,
        const aten::vec3& sun_irradiance,
        const aten::vec3& sun_direction,
        const aten::vec3& moon_direction,
        const float moon_distance)
    {
        const auto moon_radius = MoonRadius.as(MeterUnit::km);

        // Test whether the camera ray intersects the Moon sphere.
        // With the camera at the origin, the view ray is P(t) = t * v and the
        // Moon center is C = moon_distance * moon_direction. Intersections solve:
        //   |t * v - C|^2 = moon_radius^2
        // Since v is normalized, this becomes:
        //   t^2 - 2 dot(v, C) t + (dot(C, C) - moon_radius^2) = 0
        // Here dot(v, C) is projected_distance. Also, because moon_direction is
        // normalized, dot(C, C) = dot(d * m, d * m) = d^2 * dot(m, m) = d^2,
        // where d is moon_distance and m is moon_direction.
        // The usual discriminant has an extra factor of 4, which is omitted
        // because only the sign is needed for the intersection test.
        const auto view_center_cos = dot(view_direction, moon_direction);
        const auto projected_distance = view_center_cos * moon_distance;
        const auto discriminant =
            projected_distance * projected_distance
            - (moon_distance * moon_distance - moon_radius * moon_radius);

        // projected_distance <= 0 means the Moon is behind the camera ray.
        // discriminant <= 0 means this pixel is outside the visible Moon disk.
        if (projected_distance <= 0.0F || discriminant <= 0.0F) {
            return aten::vec3(0.0F);
        }

        const auto hit_distance = projected_distance - aten::sqrt(discriminant);
        const auto hit_position = view_direction * hit_distance;
        const auto moon_center = moon_direction * moon_distance;
        const auto surface_normal = normalize(hit_position - moon_center);
        const auto observer_direction = -view_direction;

        const auto cos_theta_i = aten::max(dot(surface_normal, sun_direction), 0.0F);
        const auto cos_theta_r = aten::max(dot(surface_normal, observer_direction), 0.0F);
        const auto phase_angle = aten::acos(aten::clamp(dot(sun_direction, observer_direction), -1.0F, 1.0F));

        const auto direct_brdf = ComputeMoonBrdf(cos_theta_i, cos_theta_r, phase_angle);
        const auto direct_radiance = sun_irradiance * (direct_brdf * cos_theta_i);

        // Paper Eq. (1): Earthshine irradiance at the Moon, Eem.
        // For earthshine, the incident direction is approximately the observer
        // direction, so the phase angle of the reflection is zero.
        const auto earthshine_irradiance =
            sun_irradiance * (ComputeEarthshineIrradianceAtMoon(phase_angle) / SolarIrradianceAtMoon);
        const auto earthshine_brdf = ComputeMoonBrdf(cos_theta_r, cos_theta_r, 0.0F);
        const auto earthshine_radiance = earthshine_irradiance * (earthshine_brdf * cos_theta_r);

        return direct_radiance + earthshine_radiance;
    }

    inline AT_DEVICE_API aten::vec3 RenderNightSkyBackground(
        int32_t x, int32_t y,
        const aten::CameraParameter& camera,
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::sky::PreComputeTextures& texture,
        const aten::vec3& sky_radiance_to_luminance,
        const aten::vec3& reference_irradiance,
        const aten::vec3& sun_irradiance,
        const aten::vec3& sun_direction,
        const aten::vec3& moon_direction,
        const aten::vec3& earth_center,
        const float moon_distance,
        aten::vec3& out_transmittance)
    {
        const float s = x / static_cast<float>(camera.width);
        const float t = y / static_cast<float>(camera.height);

        AT_NAME::CameraSampleResult camsample;
        AT_NAME::PinholeCamera::sample(&camsample, &camera, s, t);

        const auto camera_org{ camsample.r.org };
        const auto view_direction{ camsample.r.dir };

        constexpr float shadow_length = 0.0F;

        aten::vec3 radiance{
            GetSkyRadiance(
                atmosphere, texture,
                camera_org - earth_center,
                view_direction,
                shadow_length,
                moon_direction,
                out_transmittance)
        };

        const auto moon_irradiance = ComputeMoonIrradiance(
            sun_irradiance,
            sun_direction,
            moon_direction,
            moon_distance);
        const auto moon_irradiance_ratio = moon_irradiance / reference_irradiance;

        radiance = sky_radiance_to_luminance * (radiance * moon_irradiance_ratio);

        return radiance;
    }

    inline AT_DEVICE_API aten::vec3 RenderMoonDisk(
        int32_t x, int32_t y,
        const aten::CameraParameter& camera,
        const aten::vec3& light_radiance_to_luminance,
        const aten::vec3& sun_irradiance,
        const aten::vec3& sun_direction,
        const aten::vec3& moon_direction,
        const float moon_distance,
        const aten::vec3& transmittance)
    {
        const float s = x / static_cast<float>(camera.width);
        const float t = y / static_cast<float>(camera.height);

        AT_NAME::CameraSampleResult camsample;
        AT_NAME::PinholeCamera::sample(&camsample, &camera, s, t);

        const auto view_direction{ camsample.r.dir };

        const auto moon_radiance = GetMoonDiskRadiance(
            view_direction,
            sun_irradiance,
            sun_direction,
            moon_direction,
            moon_distance);

        return transmittance * (light_radiance_to_luminance * moon_radiance);
    }

    inline AT_DEVICE_API aten::vec3 RenderNightSky(
        int32_t x, int32_t y,
        const aten::CameraParameter& camera,
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::sky::PreComputeTextures& texture,
        const aten::vec3& light_radiance_to_luminance,
        const aten::vec3& sky_radiance_to_luminance,
        const aten::vec3& reference_irradiance,
        const aten::vec3& sun_irradiance,
        const aten::vec3& sun_direction,
        const aten::vec3& moon_direction,
        const aten::vec3& earth_center,
        const float moon_distance)
    {
        // Keep the original one-call API as a wrapper. Star rendering can use the
        // split API to compose: background -> stars -> moon disk.
        aten::vec3 transmittance{ 0.0F };
        auto radiance = RenderNightSkyBackground(
            x, y,
            camera,
            atmosphere,
            texture,
            sky_radiance_to_luminance,
            reference_irradiance,
            sun_irradiance,
            sun_direction,
            moon_direction,
            earth_center,
            moon_distance,
            transmittance);

        radiance = radiance + RenderMoonDisk(
            x, y,
            camera,
            light_radiance_to_luminance,
            sun_irradiance,
            sun_direction,
            moon_direction,
            moon_distance,
            transmittance);

        return radiance;
    }
}
