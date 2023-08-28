#pragma once

#include "defs.h"
#include "math/math.h"
#include "scene/hitable.h"
#include "scene/scene.h"
#include "scene/host_scene_context.h"
#include "camera/camera.h"
#include "misc/color.h"
#include "misc/tuple.h"
#include "sampler/sampler.h"

namespace AT_NAME
{
    class FeatureLine
    {
    private:
        FeatureLine() = delete;
        ~FeatureLine() = delete;

        FeatureLine(const FeatureLine &) = delete;
        FeatureLine(FeatureLine &&) = delete;
        FeatureLine &operator=(const FeatureLine &) = delete;
        FeatureLine &operator=(FeatureLine &&) = delete;

    public:
        /**
         * @brief Image based feature line rendering.
         * From "Ray Tracing NPR-Style Feature Lines":
         * http://www.sci.utah.edu/publications/choudhury09/NPR-lines.NPAR09.pdf
         */
        static aten::vec3 renderFeatureLine(
            const aten::vec3 &color,
            int32_t x, int32_t y,
            int32_t width, int32_t height,
            const aten::hitrecord &hrec,
            const aten::context &ctxt,
            const AT_NAME::scene &scene,
            const AT_NAME::camera &camera);

        // NOTE
        // The following functions for "Physically-based Feature Line Rendering":
        // http://lines.rexwe.st/

        /**
         * @brief Description of disc at query ray hit point.
         */
        struct Disc
        {
            aten::vec3 center;    ///< Disc center position (Query ray hit point).
            real radius{real(0)}; ///< Disc radius.

            aten::vec3 normal;            ///< Normal of disc.
            real accumulated_distance{0}; ///< Accumulated distance at disc.
        };

        /**
         * @brief Generate disc based on camera pos.
         * @note Generate disc at distance 1 from camera. Radius is that line width based screen multiplies with pixel width at distance 1 from camera.
         * @param[in] query_ray Query ray from camera.
         * @param[in] line_width Line width based on screen space.
         * @param[in] pixel_width Pixel width at distance 1 from camera.
         * @return Generated disc.
         */
        static inline AT_DEVICE_MTRL_API Disc generateDisc(
            const aten::ray &query_ray,
            real line_width,
            real pixel_width)
        {
            // Compute plane at distance 1 from camera based on camera direction.
            aten::hitrecord hrec;
            hrec.normal = -query_ray.dir;
            hrec.p = query_ray.org + query_ray.dir; // distance 1 from camera.
            const auto plane = computePlane(hrec);

            // Query ray has to hit to plane at distance 1 from camera based on camera direction.
            const auto res_pos_on_plane = computeRayHitPosOnPlane(plane, query_ray);
            const auto is_hit = aten::get<0>(res_pos_on_plane);
            const auto hit_pos_on_plane = aten::get<1>(res_pos_on_plane);
            AT_ASSERT(is_hit);

            // Hit point is the disc center.
            Disc disc;
            disc.center = hit_pos_on_plane;
            disc.normal = query_ray.dir; // First disc has to face to the same as query ray direction.
            disc.radius = line_width * pixel_width;

            return disc;
        }

        /**
         * @brief Compute disc at query ray hit point.
         * @param[in] query_ray_hit_pos Hit point of query ray.
         * @param[in] query_ray_dir Direction of query ray.
         * @param[in] previous_disc_radius Radius of previous disc.
         * @param[in] current_hit_distance Hit distance between previous query hit point and current one.
         * @param[in] accumulatedDistanceFromCameraWithout_current_hit_distance Accumulated hit distance by query ray from camera without current hit distance.
         * @return Computed disc.
         */
        static inline AT_DEVICE_MTRL_API Disc computeNextDisc(
            const aten::vec3 &query_ray_hit_pos,
            const aten::vec3 &query_ray_dir,
            real previous_disc_radius,
            real current_hit_distance,
            real accumulatedDistanceFromCameraWithout_current_hit_distance)
        {
            Disc disc;

            disc.center = query_ray_hit_pos;

            // Compute disc radius based on ratio of distance.
            const auto accumulatedDistanceFromCamera = accumulatedDistanceFromCameraWithout_current_hit_distance + current_hit_distance;
            disc.radius = previous_disc_radius * accumulatedDistanceFromCamera / accumulatedDistanceFromCameraWithout_current_hit_distance;

            // Disc normal should be opposite from query ray from 2nd disc.
            disc.normal = -query_ray_dir;

            disc.accumulated_distance = accumulatedDistanceFromCameraWithout_current_hit_distance;

            return disc;
        }

        /**
         * @brief Description how to generate sample ray.
         */
        struct SampleRayDesc
        {
            real u{0};                 ///< U on disc coordinate.
            real v{0};                 ///< V on disc coordinate.
            bool is_terminated{false}; ///< Flag if ray is termanted.
            uint8_t padding_0[3]{0, 0, 0};

            aten::vec3 prev_ray_hit_pos; ///< Hit point in previous bounce.
            real ray_org_x;              ///< Origin X of sample ray.

            aten::vec3 prev_ray_hit_nml; ///< Normal at hit point in previous bounce.
            real ray_org_y;              ///< Origin Y of sample ray.

            aten::vec3 ray_dir; ///< Direction of sample ray.
            real ray_org_z;     ///< Origin Z of sample ray.
        };

        /**
         * @brief Store sample ray to sample ray description.
         * @param desc[out] Ray description. to store sample ray.
         * @param ray[in] Ray to be stored.
         */
        static inline AT_DEVICE_MTRL_API void storeRayToDesc(
            SampleRayDesc &desc,
            const aten::ray &ray)
        {
            desc.ray_org_x = ray.org.x;
            desc.ray_org_y = ray.org.y;
            desc.ray_org_z = ray.org.z;
            desc.ray_dir = ray.dir;
        }

        /**
         * @brief Get stored ray from sample ray description.
         * @param desc[in] Sample ray description to store ray.
         * @return Stored ray.
         */
        static inline AT_DEVICE_MTRL_API aten::ray getRayFromDesc(const SampleRayDesc &desc)
        {
            aten::ray sample_ray(
                aten::vec3(desc.ray_org_x, desc.ray_org_y, desc.ray_org_z),
                desc.ray_dir);
            return sample_ray;
        }

        /**
         * @brief Generate sample ray.
         * @param[out] sample_ray_desc Description to keep how to generate sample ray.
         * @param[in] sampler Sampler to get random value.
         * @param[in] query_ray Query ray.
         * @param[in] first_disc First target disc.
         * @return Generated sample ray.
         */
        static inline AT_DEVICE_MTRL_API aten::ray generateSampleRay(
            SampleRayDesc &sample_ray_desc,
            aten::sampler &sampler,
            const aten::ray &query_ray,
            const Disc &first_disc)
        {
            const auto sample = sampler.nextSample2D();
            sample_ray_desc.u = sample.x;
            sample_ray_desc.v = sample.y;

            // [0, 1] -> [-1, 1]
            sample_ray_desc.u = sample_ray_desc.u * 2 - 1;
            sample_ray_desc.v = sample_ray_desc.v * 2 - 1;

            const auto pos_on_disc = computePosOnDisc(sample_ray_desc.u, sample_ray_desc.v, first_disc);

            // In generation timing, the origin point is the same as query ray.
            const aten::vec3 org = query_ray.org;

            const aten::vec3 ray_dir = static_cast<aten::vec3>(pos_on_disc) - org;

            aten::ray sample_ray(org, ray_dir);

            return sample_ray;
        }

        /**
         * @brief Compute next sample ray.
         * @param[in] sample_ray_desc Description of sample ray.
         * @param[in] prev_disc Previous disc at hit point of query ray.
         * @param[in] next_disc Next disc at hit point of query ray.
         * @return First variable is flag to describe if sample ray exists. Second one is next sample ray.
         *         Third one is ray target position on next disc.
         */
        static inline AT_DEVICE_MTRL_API aten::tuple<bool, aten::ray, aten::vec3> computeNextSampleRay(
            const SampleRayDesc &sample_ray_desc,
            const Disc &prev_disc,
            const Disc &next_disc)
        {
            // NOTE:
            // If we detect feature line, everything stop.
            // It means every ray (query & sample) traversal is terminated.
            // So, we don't need to check if sample ray is terminated individually.

            // Position on next disc.
            const auto disc_face_ratio = dot(prev_disc.normal, next_disc.normal);
            const auto u = disc_face_ratio >= 0 ? sample_ray_desc.u : -sample_ray_desc.u;
            const auto v = sample_ray_desc.v;
            const auto pos_on_next_disc = computePosOnDisc(u, v, next_disc);

            const auto &prev_org = sample_ray_desc.prev_ray_hit_pos;

            // Next sample ray direction.
            auto ray_dir = static_cast<aten::vec3>(pos_on_next_disc) - prev_org;

            ray_dir = normalize(ray_dir);
            const auto d = dot(ray_dir, sample_ray_desc.prev_ray_hit_nml);
            if (d < 0)
            {
                // Not allow bounce to the different direction from normal.
                return aten::make_tuple<bool, aten::ray, aten::vec3>(false, aten::ray(), aten::vec3());
            }

            // Normal is necessary to avoid self hit.
            //   This normal is at sample ray hit position in previous bounce.
            //   "sample ray hit position in previous bounce." = origin point of next sample ray.
            aten::ray next_sample_ray(prev_org, ray_dir, sample_ray_desc.prev_ray_hit_nml);
            if (aten::isInvalid(next_sample_ray.dir))
            {
                return aten::make_tuple<bool, aten::ray, aten::vec3>(false, aten::ray(), aten::vec3());
            }

            return aten::make_tuple<bool, aten::ray, aten::vec3>(true, next_sample_ray, pos_on_next_disc);
        }

        /**
         * @brief Compute hit position by ray on disc.
         * @param[in] u U on disc coordinate.
         * @param[in] v V on disc coordinate.
         * @param[in] disc Disc which ray aims to hit.
         * @return Hit position by ray on disc.
         */
        static inline AT_DEVICE_MTRL_API aten::vec4 computePosOnDisc(
            const real u, const real v,
            const Disc &disc)
        {
            // theta 0 is based on X-axis.
            aten::vec4 pos_on_next_disc(u, v, 0, 1);

            // Adjust with disc radius.
            pos_on_next_disc *= disc.radius;

            // Coordinate based on normal of next disc.
            const auto n = disc.normal;
            const auto t = aten::getOrthoVector(n);
            const auto b = cross(n, t);

            // Make matrix from n, t, b
            aten::mat4 mtx_axes(t, b, n);
            pos_on_next_disc = mtx_axes.applyXYZ(pos_on_next_disc);

            // Trans
            pos_on_next_disc.x += disc.center.x;
            pos_on_next_disc.y += disc.center.y;
            pos_on_next_disc.z += disc.center.z;

            return pos_on_next_disc;
        }

        /**
         * @brief Compute plane from hit record.
         * @param[in] hrec Hit record to compute plane.
         * @return Computed plane as vec4 <Normal of plane, D>.
         */
        static inline AT_DEVICE_MTRL_API aten::vec4 computePlane(const aten::hitrecord &hrec)
        {
            // NOTE:
            // Plane:
            // ax + by + cz + d = 0
            // normal = (x, y, z), p_on_plane = (a, b, c)
            const auto &n = hrec.normal;
            const auto &p = hrec.p;
            auto d = -dot(n, p);

            return aten::vec4(n.x, n.y, n.z, d);
        }

        /**
         * @brief Compute ray hit position on plane.
         * @param[in] plane Plane as vec4 <Normal of plane, D>.
         * @param[in] ray Ray.
         * @return First variable is flag to describe if ray hits to plane. Second one is hit position on plane.
         */
        static inline AT_DEVICE_MTRL_API aten::tuple<bool, aten::vec3> computeRayHitPosOnPlane(
            const aten::vec4 &plane,
            const aten::ray &ray)
        {
            // NOTE:
            // Plane : L = <N, D>
            // Ray   : P(t) = Q + Vt
            //
            // NP(t) + D = 0
            //   <=> N(Q + Vt) + D = 0
            //   <=> NQ + (NV)t + D = 0
            //   <=> t = -(NQ + D) / NV
            //   <=> t = -LQ / LV
            //
            // Q = (Qx, Qy, Qz, 1) => Ray org
            // V = (Vx, Vy, Vz, 0) => Ray dir
            // L = (Nx, Ny, Nz, D) => Plane
            // Therefore:
            //   dot(N, Q) + D = Nx * Qx + Ny * Qy + Nz * Qz + D * 1 = LQ
            //   dot(N, V) + 0 = Nx * Vx + Ny * Vy + Nz * Vz + D * 0 = LV

            const auto& L = plane;
            const aten::vec4 Q(ray.org, 1);
            const aten::vec4 V(ray.dir, 0);

            // LV
            const auto div = dot(L, V);
            if (div == 0)
            {
                return aten::make_tuple<bool, aten::vec3>(false, aten::vec3());
            }

            // t = -LQ / LV
            auto t = dot(L, Q);
            t = -t / div;

            aten::vec3 pos = ray.org + t * ray.dir;
            return aten::make_tuple<bool, aten::vec3>(t >= 0, pos);
        }

        /**
         * @brief Compute distance between point and ray.
         * @param[in] point Point to compute distance to ray.
         * @param[in] ray Ray to compute distance with point.
         * @param[out] hit_point Storage to get projected point on ray. If this value is null, point can't be stored.
         * @return Distance between point and ray.
         */
        static inline AT_DEVICE_MTRL_API real computeDistanceBetweenPointAndRay(
            const aten::vec3 &point,
            const aten::ray &ray,
            aten::vec3 *hit_point)
        {
            // NOTE
            // https://ja.wikipedia.org/wiki/%E7%82%B9%E3%81%A8%E7%9B%B4%E7%B7%9A%E3%81%AE%E8%B7%9D%E9%9B%A2#%E3%83%99%E3%82%AF%E3%83%88%E3%83%AB%E3%82%92%E7%94%A8%E3%81%84%E3%81%9F%E5%85%AC%E5%BC%8F
            // ray: R = O + tV
            // point: P
            // Compute vector from ray org to point: X = P - O
            // Compute project X to R: Y = dot(P - O, V) * V
            // Y is the shortest hit point on ray from point.
            // So, we can get distance from point to ray with computing distance between P and Y.
            // distance = length(P - Y)

            const auto X = point - ray.org;

            // Compute projected point.
            auto Y = dot(X, ray.dir) * ray.dir;
            Y += ray.org;

            if (hit_point)
            {
                *hit_point = Y;
            }

            const auto distance = length(point - Y);

            return distance;
        }

        /**
         * @brief Compute distance between projected point on ray and origin of ray.
         * @param point[in] Point to be projected on ray.
         * @param ray[in] Ray which point is projected on.
         * @return Dstance between projected point on ray and origin of ray.
         */
        static inline AT_DEVICE_MTRL_API real computeDistanceBetweenProjectedPosOnRayAndRayOrg(
            const aten::vec3 &point,
            const aten::ray &ray)
        {
            aten::vec3 sample_pos_on_query_ray;
            (void)computeDistanceBetweenPointAndRay(point, ray, &sample_pos_on_query_ray);
            const auto distance_sample_pos_on_ray = length(sample_pos_on_query_ray - ray.org);
            return distance_sample_pos_on_ray;
        }

        /**
         * @brief Evaluate feature line metrics.
         * @param[in] p Start point of query ray.
         * @param[in] scale_factor Scale factor for depth threshold.
         * @param[in] hrec_query Hit record of query ray.
         * @param[in] hrec_sample Hit record of sample ray.
         * @param[in] albedo_q Albedo at end point of query ray.
         * @param[in] albedo_s Albedo at sample ray hit point.
         * @param[in] depth_q Depth at end point of query ray.
         * @param[in] depth_s Depth at sample ray hit point.
         * @param[in] threshold_albedo Threshold for albedo.
         * @param[in] threshold_normal Threshold for normal.
         * @param[in] scale_factor_threshold_depth Scale factor to compute depth threshold.
         * @return If metric is valid, return true. Otherwise, return false.
         */
        static inline AT_DEVICE_MTRL_API bool evaluateMetrics(
            const aten::vec3 &p,
            const aten::hitrecord &hrec_query,
            const aten::hitrecord &hrec_sample,
            const aten::vec4 &albedo_q,
            const aten::vec4 &albedo_s,
            const real depth_q,
            const real depth_s,
            const real threshold_albedo,
            const real threshold_normal,
            const real scale_factor_threshold_depth)
        {
            // NOTE
            // Combine metrics by taking their max term (i.e. if any metric would return true the combined metric returns true).

            // Mesh id.
            const auto is_mesh = evaluateMeshIdMetric(hrec_query.meshid, hrec_sample.meshid);

            // Albedo.
            const auto is_albedo = evaluateAlbedoMetric(threshold_albedo, albedo_q, albedo_s);

            // Normal.
            const auto is_normal = evaluateNormalMetric(threshold_normal, hrec_query.normal, hrec_sample.normal);

            // Depth
            const auto is_depth = evaluateDepthMetric(
                p, scale_factor_threshold_depth, hrec_query, hrec_sample, depth_q, depth_s);

            return is_mesh || is_albedo || is_normal || is_depth;
        }

        /**
         * @brief Evaluate feature line metric for mesh id.
         * @param[in] mesh_id_query Mesh id at end point of query ray.
         * @param[in] hrec_sample Mesh id at sample ray hit point.
         * @return If metric is valid, return true. Otherwise, return false.
         */
        static inline AT_DEVICE_MTRL_API bool evaluateMeshIdMetric(
            const int32_t mesh_id_query,
            const int32_t mesh_id_sample)
        {
            const auto is_mesh = (mesh_id_query != mesh_id_sample);
            return is_mesh;
        }

        /**
         * @brief Evaluate feature line metric for albedo.
         * @param[in] threshold_albedo Threshold for albedo.
         * @param[in] albedo_q Albedo at end point of query ray.
         * @param[in] albedo_s Albedo at sample ray hit point.
         * @return If metric is valid, return true. Otherwise, return false.
         */
        static inline AT_DEVICE_MTRL_API bool evaluateAlbedoMetric(
            const real threshold_albedo,
            const aten::vec4 &albedo_q,
            const aten::vec4 &albedo_s)
        {
            const auto lum_q = AT_NAME::color::luminance(albedo_q);
            const auto lum_s = AT_NAME::color::luminance(albedo_s);
            const auto is_albedo = aten::abs(lum_q - lum_s) > threshold_albedo;
            return is_albedo;
        }

        /**
         * @brief Evaluate feature line metric for normal.
         * @param[in] threshold_normal Threshold for normal.
         * @param[in] normal_query Albedo at end point of query ray.
         * @param[in] normal_sample Albedo at sample ray hit point.
         * @return If metric is valid, return true. Otherwise, return false.
         */
        static inline AT_DEVICE_MTRL_API bool evaluateNormalMetric(
            const real threshold_normal,
            const aten::vec3 &normal_query,
            const aten::vec3 &normal_sample)
        {
            const auto is_normal = (real(1) - dot(normal_query, normal_sample)) > threshold_normal;
            return is_normal;
        }

        /**
         * @brief Evaluate feature line metric for depth.
         * @param[in] p Start point of query ray.
         * @param[in] scale_factor Scale factor for depth threshold.
         * @param[in] hrec_query Hit record of query ray.
         * @param[in] hrec_sample Hit record of sample ray.
         * @param[in] depth_q Depth at end point of query ray.
         * @param[in] depth_s Depth at sample ray hit point.
         * @return If metric is valid, return true. Otherwise, return false.
         */
        static inline AT_DEVICE_MTRL_API bool evaluateDepthMetric(
            const aten::vec3 &p,
            const real scale_factor,
            const aten::hitrecord &hrec_query,
            const aten::hitrecord &hrec_sample,
            const real depth_q,
            const real depth_s)
        {
            const auto threshold_depth = computeThresholdDepth(
                p,
                scale_factor,
                hrec_query, hrec_sample,
                depth_q, depth_s);
            const auto is_depth = aten::abs(depth_q - depth_s) > threshold_depth;
            return is_depth;
        }

        /**
         * @brief Compute threshold for depth.
         * @param[in] p Start point of query ray.
         * @param[in] scale_factor Scale factor for depth threshold.
         * @param[in] hrec_query Hit record of query ray.
         * @param[in] hrec_sample Hit record of sample ray.
         * @param[in] depth_q Depth at end point of query ray.
         * @param[in] depth_s Depth at sample ray hit point.
         * @return Threshold for depth.
         */
        static inline AT_DEVICE_MTRL_API real computeThresholdDepth(
            const aten::vec3 &p,
            const real scale_factor,
            const aten::hitrecord &hrec_query,
            const aten::hitrecord &hrec_sample,
            const real depth_q,
            const real depth_s)
        {
            // NOTE:
            // t_depth = b * max(dq, ds) * lengh(ps - pq) / abs(dot(pq, n_closest)
            //  b : scaling factor
            //  p : start point of query ray
            //  q : end point of query ray
            //  s : sample ray hit point
            //  dq : depth at point q
            //  ds : depth at point s
            //  ps : vector p -> s
            //  pq : vector p -> q
            //  n_closest : normal at q or s which is closer to p

            const auto p_q = hrec_query.p - p;
            const auto p_s = hrec_sample.p - p;

            const auto &n_closest = length(p_q) > length(p_s) ? hrec_sample.normal : hrec_query.normal;

            const auto max_depth = std::max(depth_q, depth_s);
            const auto div = aten::abs(dot(p_q, n_closest));

            if (div == real(0))
            {
                return FLT_MAX;
            }

            const auto t_depth = scale_factor * max_depth * length(p_s - p_q) / div;

            return t_depth;
        }

        /**
         * @brief Check if feature line width in 3D is valid based on line width in 2D.
         * @param[in] screen_line_width Line width in screen space (2D).
         * @param[in] query_ray Query ray.
         * @param[in] sample_hit_point Hit point of sample ray.
         * @param[in] accumulatedDistance Accumulated distance at feature line point on ray from camera.
         * @param[in] pixelWidth Pixel width at distance 1 from camera.
         * @return If feature line width in 3D is valid based on line width in 2D, return true. Otherwise, return false.
         */
        static inline AT_DEVICE_MTRL_API bool isInLineWidth(
            const real screen_line_width,
            const aten::ray &query_ray,
            const aten::vec3 &sample_hit_point,
            const real accumulatedDistance,
            const real pixelWidth)
        {
            // NOTE:
            // Wscaled = d * Pw * Wscreen
            //  Wscaled : Distance between query ray and feature line point.
            //  Wscreen : Line width as pixel size.
            //  d : Accumulated distance at feature line point on ray from camera.
            //  Pw: Pixel width at distance 1 from camera.

            aten::vec3 hit_point_on_ray;
            const auto length_point_ray = computeDistanceBetweenPointAndRay(sample_hit_point, query_ray, &hit_point_on_ray);

            auto distance = length(query_ray.org - hit_point_on_ray);
            distance = accumulatedDistance + distance;

            const auto w_scaled = distance * pixelWidth * screen_line_width;

            const auto is_in_line_width = length_point_ray <= w_scaled;
            return is_in_line_width;
        }
    };
}
