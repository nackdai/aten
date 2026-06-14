#pragma once

#include "atmosphere/sky/star_types.h"

#include "camera/camera.h"

#include "math/math.h"
#include "math/vec2.h"
#include "math/vec3.h"

namespace aten::sky {
    /**
     * @brief Parameters controlling point-star screen splats.
     *
     * Star brightness comes from catalog magnitude and is stored in Star::irradiance.
     * These parameters control only the render footprint, debug/exposure scale, and
     * horizon culling behavior.
     */
    struct StarRenderParameters {
        // Render-footprint angular radius, i.e. the apparent radius used only for
        // drawing the star point. This is not the physical apparent radius of the
        // stellar disk: real stellar disks are far below a pixel, so brightness is
        // kept in Star::irradiance while visible size is controlled separately.
        float angular_radius{ aten::Deg2Rad(0.03F) };

        // Soft edge width around angular_radius. This reduces hard one-pixel
        // popping before a later screen-space/star splat implementation exists.
        float angular_fade_width{ aten::Deg2Rad(0.01F) };

        // Radiance multiplier for exposure/debug tuning. Physical brightness is
        // still sourced from Star::irradiance.
        float radiance_scale{ 1.0F };

        // If true, directions below the local horizon are rejected. In that case
        // star_direction must already be expressed in the local frame whose +Y is
        // the local zenith.
        bool cull_below_horizon{ true };
    };

    /**
     * @brief Screen-space representation of one projected star.
     *
     * This is produced per star by BuildStarScreenSplat and is intended to be
     * scattered/splatted to only the pixels covered by the footprint. This avoids
     * the expensive "each pixel scans all stars" approach.
     */
    struct StarScreenSplat {
        // Pixel-space center of the star footprint.
        aten::vec2 center{ 0.0F, 0.0F };

        // Inner radius and soft edge width in pixels.
        float radius{ 0.0F };
        float fade_width{ 0.0F };

        // Observed star radiance. This already includes catalog irradiance,
        // luminance-normalized RGB weight, atmospheric transmittance, and caller-
        // controlled radiance_scale.
        aten::vec3 radiance{ 0.0F };

        // Original catalog id. Useful for debug picking and later filtering.
        uint32_t hr{ 0 };
    };

    /**
     * @brief Return whether a local-frame star direction is above the horizon.
     *
     * @param star_direction Star direction in the local horizon frame where +Y is zenith.
     * @return true if the star is above the local horizon.
     */
    inline AT_HOST_DEVICE_API bool IsStarAboveHorizon(const aten::vec3& star_direction)
    {
        // Local sky convention used in the night-sky notes: +Y is local zenith.
        return star_direction.y > 0.0F;
    }

    /**
     * @brief Compute angular-kernel weight for a star along one view direction.
     *
     * This helper is useful for tests and ray-style evaluation of a single star.
     * The production-oriented path should prefer BuildStarScreenSplat plus
     * EvaluateStarSplatRadiance so rendering can be star-centric instead of
     * pixel-centric.
     *
     * @param view_direction Normalized view ray direction.
     * @param star_direction Normalized star direction in the same frame as view_direction.
     * @param angular_radius Inner angular radius of the render footprint.
     * @param angular_fade_width Soft fade width outside angular_radius.
     * @return Weight in [0, 1].
     */
    inline AT_HOST_DEVICE_API float ComputeStarAngularWeight(
        const aten::vec3& view_direction,
        const aten::vec3& star_direction,
        const float angular_radius,
        const float angular_fade_width)
    {
        const auto cos_theta = aten::clamp(dot(view_direction, star_direction), -1.0F, 1.0F);

        // Compare cosines instead of calling acos for the common outside test.
        // For small angles, cos(theta) is larger when theta is closer to zero.
        const auto cos_outer = aten::cos(angular_radius + aten::max(angular_fade_width, 0.0F));
        if (cos_theta <= cos_outer) {
            return 0.0F;
        }

        const auto cos_inner = aten::cos(angular_radius);
        if (cos_theta >= cos_inner || angular_fade_width <= 0.0F) {
            return 1.0F;
        }

        return aten::smoothstep(cos_outer, cos_inner, cos_theta);
    }

    /**
     * @brief Compute the star's RGB radiance/irradiance contribution before atmosphere.
     *
     * Star::irradiance is a scalar value derived from Vmag, and
     * Star::rgb_irradiance_weight distributes that scalar into RGB while preserving
     * luminance. Atmospheric transmittance is not applied here because it depends
     * on the observer/view path; BuildStarScreenSplat applies it when producing the
     * observed screen splat.
     *
     * @param star Renderer-ready star data loaded from the catalog.
     * @param radiance_scale Caller-controlled scale for exposure/debug tuning.
     * @return RGB star contribution before atmospheric transmittance.
     */
    inline AT_HOST_DEVICE_API aten::vec3 ComputeStarRadiance(
        const Star& star,
        const float radiance_scale)
    {
        // Star::irradiance is already derived from Vmag at catalog-load time.
        // Star::rgb_irradiance_weight is derived from B-V/Teff and normalized to
        // linear-sRGB luminance 1, so multiplying it by the scalar Vmag-derived
        // irradiance preserves the brightness while distributing it into RGB.
        // Atmospheric transmittance is intentionally not applied here; it depends
        // on the observer/view path and is handled by the caller when building the
        // observed screen splat.
        return star.rgb_irradiance_weight * (star.irradiance * radiance_scale);
    }

    /**
     * @brief Project an infinite-distance star direction to pixel coordinates.
     *
     * Uses the same pinhole camera convention as PinholeCamera::RevertRayToPixelPos.
     * The star is treated as infinitely far away, so only direction is needed.
     *
     * @param camera Pinhole camera parameters.
     * @param star_direction Normalized star direction in the camera/world frame.
     * @param screen_position Output pixel-space position.
     * @return true if the projected position is inside the camera viewport.
     */
    inline AT_HOST_DEVICE_API bool ProjectStarToScreen(
        const aten::CameraParameter& camera,
        const aten::vec3& star_direction,
        aten::vec2& screen_position)
    {
        // Same pinhole projection convention as PinholeCamera::RevertRayToPixelPos.
        // The star is treated as being at infinity, so only its direction matters.
        const auto forward = dot(star_direction, camera.dir);
        if (forward <= 0.0F) {
            return false;
        }

        const auto distance_to_screen = camera.dist / forward;
        const auto screen_pos = camera.origin + star_direction * distance_to_screen - camera.center;

        screen_position.x = dot(screen_pos, camera.right) + static_cast<float>(camera.width) * 0.5F;
        screen_position.y = dot(screen_pos, camera.up) + static_cast<float>(camera.height) * 0.5F;

        return screen_position.x >= 0.0F
            && screen_position.x < static_cast<float>(camera.width)
            && screen_position.y >= 0.0F
            && screen_position.y < static_cast<float>(camera.height);
    }

    /**
     * @brief Convert a render-footprint apparent radius to pixel radius.
     *
     * @param camera Pinhole camera parameters. camera.dist must be in pixel units.
     * @param angular_radius Apparent angular radius of the render footprint.
     * @return Screen-space radius in pixels.
     */
    inline AT_HOST_DEVICE_API float ComputeStarAngularRadiusInPixels(
        const aten::CameraParameter& camera,
        const float angular_radius)
    {
        // Convert the render-footprint apparent radius to a pixel radius.
        //
        // Think of the camera ray and the image plane as a right triangle:
        //
        //   camera
        //     o
        //     |\
        //   d | \  theta = angular_radius
        //     |  \
        //     +---*
        //       r
        //
        // theta is the apparent angular radius, d is camera.dist, and r is the
        // radius on the image plane. PinholeCamera stores camera.dist in pixel
        // units:
        //   camera.dist = image_height / (2 * tan(vertical_fov / 2)).
        // Therefore the screen-space radius is:
        //   tan(theta) = r / d  =>  r = tan(theta) * d.
        //
        // For stars this theta is only a render-footprint apparent radius. Real
        // stellar disks are far below a pixel, so their brightness comes from
        // Star::irradiance while visible size is controlled separately here.
        return aten::tan(angular_radius) * camera.dist;
    }

    /**
     * @brief Build the screen-space splat for one star.
     *
     * The caller provides the observation-dependent star direction and atmospheric
     * transmittance. This function handles horizon culling, camera projection,
     * footprint conversion, and observed radiance construction. It does not write
     * to an image; callers should splat the returned StarScreenSplat over its
     * covered pixels.
     *
     * @param star Renderer-ready catalog star.
     * @param star_direction Observation-dependent star direction in the local/camera frame.
     * @param transmittance Atmospheric transmittance from top of atmosphere to camera.
     * @param camera Pinhole camera parameters.
     * @param params Star rendering parameters.
     * @param splat Output screen-space splat.
     * @return true if the star should produce a visible splat.
     */
    inline AT_HOST_DEVICE_API bool BuildStarScreenSplat(
        const Star& star,
        const aten::vec3& star_direction,
        const aten::vec3& transmittance,
        const aten::CameraParameter& camera,
        const StarRenderParameters& params,
        StarScreenSplat& splat)
    {
        if (params.cull_below_horizon && !IsStarAboveHorizon(star_direction)) {
            return false;
        }

        if (!ProjectStarToScreen(camera, star_direction, splat.center)) {
            return false;
        }

        splat.radius = ComputeStarAngularRadiusInPixels(camera, params.angular_radius);
        splat.fade_width = ComputeStarAngularRadiusInPixels(camera, params.angular_fade_width);
        const auto star_radiance = ComputeStarRadiance(star, params.radiance_scale);
        splat.radiance = transmittance * star_radiance;
        splat.hr = star.hr;

        return true;
    }

    /**
     * @brief Compute the weight of a screen-space splat at a pixel position.
     *
     * @param splat Star splat produced by BuildStarScreenSplat.
     * @param pixel_x Pixel x coordinate, usually including a 0.5 center offset.
     * @param pixel_y Pixel y coordinate, usually including a 0.5 center offset.
     * @return Weight in [0, 1].
     */
    inline AT_HOST_DEVICE_API float ComputeStarSplatWeight(
        const StarScreenSplat& splat,
        const float pixel_x,
        const float pixel_y)
    {
        const auto dx = pixel_x - splat.center.x;
        const auto dy = pixel_y - splat.center.y;
        const auto distance = aten::sqrt(dx * dx + dy * dy);

        const auto outer_radius = splat.radius + aten::max(splat.fade_width, 0.0F);
        if (distance >= outer_radius) {
            return 0.0F;
        }

        if (distance <= splat.radius || splat.fade_width <= 0.0F) {
            return 1.0F;
        }

        return 1.0F - aten::smoothstep(splat.radius, outer_radius, distance);
    }

    /**
     * @brief Evaluate the observed radiance contributed by one splat at one pixel.
     *
     * @param splat Star splat produced by BuildStarScreenSplat.
     * @param pixel_x Pixel x coordinate, usually including a 0.5 center offset.
     * @param pixel_y Pixel y coordinate, usually including a 0.5 center offset.
     * @return RGB radiance contribution for the pixel.
     */
    inline AT_HOST_DEVICE_API aten::vec3 EvaluateStarSplatRadiance(
        const StarScreenSplat& splat,
        const float pixel_x,
        const float pixel_y)
    {
        return splat.radiance * ComputeStarSplatWeight(splat, pixel_x, pixel_y);
    }
}
