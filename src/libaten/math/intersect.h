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
        real a, b;

        real t;
    };

    inline AT_DEVICE_API int maxDim(const vec3& v)
    {
        uint32_t x = (uint32_t)aten::abs(v.x);
        uint32_t y = (uint32_t)aten::abs(v.y);
        uint32_t z = (uint32_t)aten::abs(v.z);

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

    inline intersectResult intersectTriangle(
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

        real inv = real(1) / dot(u, e1);

        real t = dot(v, e2) * inv;
        real beta = dot(u, r) * inv;
        real gamma = dot(v, d) * inv;

        intersectResult result;

        result.isIntersect = ((beta >= real(0) && beta <= real(1))
            && (gamma >= real(0) && gamma <= real(1))
            && (beta + gamma <= real(1))
            && t >= real(0));

        result.a = beta;
        result.b = gamma;

        result.t = t;

        return std::move(result);
#else
        intersectResult result;

        // NOTE
        // Watertight Ray/Triangle Intersection
        // http://jcgt.org/published/0002/01/05/paper.pdf

        // calculate dimension where ray direction is maximal.
        int kz = maxDim(ray.dir);
        int kx = (kz + 1) % 3;
        int ky = (kx + 1) % 3;

        // swap kx and ky dimension to preserve windin direction of triangles.
        if (ray.dir[kz] < real(0)) {
            int tmp = kx;
            kx = ky;
            ky = tmp;
        }

        // calculate shear constants.
        real Sx = ray.dir[kx] / ray.dir[kz];
        real Sy = ray.dir[ky] / ray.dir[kz];
        real Sz = real(1) / ray.dir[kz];

        // calculate vertices relative to ray origin.
        const auto A = v0 - ray.org;
        const auto B = v1 - ray.org;
        const auto C = v2 - ray.org;

        // perform shear and scale of vertices.
        const real Ax = A[kx] - Sx * A[kz];
        const real Ay = A[ky] - Sy * A[kz];
        const real Bx = B[kx] - Sx * B[kz];
        const real By = B[ky] - Sy * B[kz];
        const real Cx = C[kx] - Sx * C[kz];
        const real Cy = C[ky] - Sy * C[kz];

        // calculate scaled barycentric coordinates.
        real U = Cx * By - Cy * Bx;
        real V = Ax * Cy - Ay * Cx;
        real W = Bx * Ay - By * Ax;

        // Peform edge tests.
        // Moving this test before
        // and the end of the previous conditional gives higher performance.
        if ((U < real(0) || V < real(0) || W < real(0))
            && (U > real(0) || V > real(0) || W > real(0)))
        {
            return std::move(result);
        }

        // calculate dterminant.
        real det = U + V + W;

        if (det == real(0)) {
            return std::move(result);
        }

        // Calculate scaled z-coordinated of vertice
        // and use them to calculate the hit distance.
        const real Az = Sz * A[kz];
        const real Bz = Sz * B[kz];
        const real Cz = Sz * C[kz];
        const real T = U * Az + V * Bz + W * Cz;

        const real rcpDet = real(1) / det;

        const real beta = U * rcpDet;
        const real gamma = V * rcpDet;
        const real t = T * rcpDet;

        result.isIntersect = ((beta >= real(0) && beta <= real(1))
            && (gamma >= real(0) && gamma <= real(1))
            && (beta + gamma <= real(1))
            && t >= real(0));

        result.a = beta;
        result.b = gamma;
        result.t = t;

        return std::move(result);
#endif
    }
}
