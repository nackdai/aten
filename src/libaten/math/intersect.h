#pragma once

#include "defs.h"
#include "math/vec3.h"
#include "math/ray.h"

namespace aten {
    struct intersectResult {
        bool isIntersect{ false };

        // NOTE
        // http://d.hatena.ne.jp/Zellij/20131207/p1

        // 重心座標系(barycentric coordinates).
        // v0基準.
        // p = (1 - a - b)*v0 + a*v1 + b*v2
        float a{ 0.0F };
        float b{ 0.0F };

        float t{ 0.0F };
    };

    inline AT_HOST_DEVICE_API int32_t maxDim(const vec3& v)
    {
        const auto x = static_cast<int32_t>(aten::abs(v.x));
        const auto y = static_cast<int32_t>(aten::abs(v.y));
        const auto z = static_cast<int32_t>(aten::abs(v.z));

        if (x > y) {
            if (x > z) {
                return 0;
            }
            else {
                return 2;
            }
        }
        else if (y > z) {
            return 1;
        }
        else {
            return 2;
        }
    }

    inline AT_HOST_DEVICE_API intersectResult intersectTriangle(
        const ray& ray,
        const vec3& v0,
        const vec3& v1,
        const vec3& v2)
    {
        // NOTE
        // http://qiita.com/edo_m18/items/2bd885b13bd74803a368
        // http://kanamori.cs.tsukuba.ac.jp/jikken/inner/triangle_intersection.pdf
        // https://pheema.hatenablog.jp/entry/ray-triangle-intersection

        // Fast Minimum Storage Ray Triangle Intersection.
        // https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf

#if 1
        vec3 e1 = v1 - v0;
        vec3 e2 = v2 - v0;
        vec3 r = ray.org - v0;
        vec3 d = ray.dir;

        // NOTE
        // To avoid 'if' for GPU computing, I don't run 'if' confition to check 'u', 'v' here.

        vec3 u = cross(d, e2);
        vec3 v = cross(r, e1);

        float inv = float(1) / dot(u, e1);

        float t = dot(v, e2) * inv;
        float beta = dot(u, r) * inv;
        float gamma = dot(v, d) * inv;

        intersectResult result;

        result.isIntersect = ((beta >= float(0) && beta <= float(1))
            && (gamma >= float(0) && gamma <= float(1))
            && (beta + gamma <= float(1))
            && t >= float(0));

        result.a = beta;
        result.b = gamma;

        result.t = t;

        return result;;
#else
        intersectResult result;

        // NOTE
        // Watertight Ray/Triangle Intersection
        // http://jcgt.org/published/0002/01/05/paper.pdf

        // calculate dimension where ray direction is maximal.
        int32_t kz = maxDim(ray.dir);
        int32_t kx = (kz + 1) % 3;
        int32_t ky = (kx + 1) % 3;

        // swap kx and ky dimension to preserve windin direction of triangles.
        if (ray.dir[kz] < float(0)) {
            int32_t tmp = kx;
            kx = ky;
            ky = tmp;
        }

        // calculate shear constants.
        float Sx = ray.dir[kx] / ray.dir[kz];
        float Sy = ray.dir[ky] / ray.dir[kz];
        float Sz = float(1) / ray.dir[kz];

        // calculate vertices relative to ray origin.
        const auto A = v0 - ray.org;
        const auto B = v1 - ray.org;
        const auto C = v2 - ray.org;

        // perform shear and scale of vertices.
        const float Ax = A[kx] - Sx * A[kz];
        const float Ay = A[ky] - Sy * A[kz];
        const float Bx = B[kx] - Sx * B[kz];
        const float By = B[ky] - Sy * B[kz];
        const float Cx = C[kx] - Sx * C[kz];
        const float Cy = C[ky] - Sy * C[kz];

        // calculate scaled barycentric coordinates.
        float U = Cx * By - Cy * Bx;
        float V = Ax * Cy - Ay * Cx;
        float W = Bx * Ay - By * Ax;

        // Peform edge tests.
        // Moving this test before
        // and the end of the previous conditional gives higher performance.
        if ((U < float(0) || V < float(0) || W < float(0))
            && (U > float(0) || V > float(0) || W > float(0)))
        {
            return result;;
        }

        // calculate dterminant.
        float det = U + V + W;

        if (det == float(0)) {
            return result;;
        }

        // Calculate scaled z-coordinated of vertice
        // and use them to calculate the hit distance.
        const float Az = Sz * A[kz];
        const float Bz = Sz * B[kz];
        const float Cz = Sz * C[kz];
        const float T = U * Az + V * Bz + W * Cz;

        const float rcpDet = float(1) / det;

        const float beta = U * rcpDet;
        const float gamma = V * rcpDet;
        const float t = T * rcpDet;

        result.isIntersect = ((beta >= float(0) && beta <= float(1))
            && (gamma >= float(0) && gamma <= float(1))
            && (beta + gamma <= float(1))
            && t >= float(0));

        result.a = beta;
        result.b = gamma;
        result.t = t;

        return result;;
#endif
    }
}
