#include "defs.h"

#include "geometry/EvaluateHitResult.h"

// Compare Less EQual
inline __device__ int32_t cmpLEQ(const float4& a, const float4& b)
{
    aten::_vec4_cmp_res res;

    res.f = 0;
    res._0 = (a.x <= b.x);
    res._1 = (a.y <= b.y);
    res._2 = (a.z <= b.z);
    res._3 = (a.w <= b.w);

    return res.f;
}

// Compare Greater EQual
inline __device__ int32_t cmpGEQ(const float4& a, const float4& b)
{
    aten::_vec4_cmp_res res;

    res.f = 0;
    res._0 = (a.x >= b.x);
    res._1 = (a.y >= b.y);
    res._2 = (a.z >= b.z);
    res._3 = (a.w >= b.w);

    return res.f;
}

AT_INLINE_RELEASE __device__ int32_t hit4Triangles1Ray(
    const idaten::context* ctxt,
    float4 primIdx, int32_t num,
    float4* resultT,
    float4* resultA,
    float4* resultB,
    aten::vec3 org, aten::vec3 dir,
    float4 v0x, float4 v0y, float4 v0z,
    float4 e1x, float4 e1y, float4 e1z)
{
    float4 v2x, v2y, v2z;

    for (int32_t i = 0; i < num; i++) {
        int32_t pidx = (int32_t)*((float*)&primIdx + i);
        const auto* prim = &ctxt->prims[pidx];
        float4 p2{ ctxt->GetPosition(prim->idx[2]) };

        *(((float*)&v2x) + i) = p2.x;
        *(((float*)&v2y) + i) = p2.y;
        *(((float*)&v2z) + i) = p2.z;
    }

    // e1 = v1 - v0
    const auto& e1_x = e1x;
    const auto& e1_y = e1y;
    const auto& e1_z = e1z;

    // e2 = v2 - v0
    auto e2_x = v2x - v0x;
    auto e2_y = v2y - v0y;
    auto e2_z = v2z - v0z;

    float4 ox = make_float4(org.x, org.x, org.x, org.x);
    float4 oy = make_float4(org.y, org.y, org.y, org.y);
    float4 oz = make_float4(org.z, org.z, org.z, org.z);

    // d
    float4 d_x = make_float4(dir.x, dir.x, dir.x, dir.x);
    float4 d_y = make_float4(dir.y, dir.y, dir.y, dir.y);
    float4 d_z = make_float4(dir.z, dir.z, dir.z, dir.z);

    // r = r.org - v0
    auto r_x = ox - v0x;
    auto r_y = oy - v0y;
    auto r_z = oz - v0z;

    // u = cross(d, e2)
    auto u_x = d_y * e2_z - d_z * e2_y;
    auto u_y = d_z * e2_x - d_x * e2_z;
    auto u_z = d_x * e2_y - d_y * e2_x;

    // v = cross(r, e1)
    auto v_x = r_y * e1_z - r_z * e1_y;
    auto v_y = r_z * e1_x - r_x * e1_z;
    auto v_z = r_x * e1_y - r_y * e1_x;

    // inv = real(1) / dot(u, e1)
    auto divisor = u_x * e1_x + u_y * e1_y + u_z * e1_z;
    auto inv = real(1) / (divisor + real(1e-6));

    // t = dot(v, e2) * inv
    auto t = (v_x * e2_x + v_y * e2_y + v_z * e2_z) * inv;

    // beta = dot(u, r) * inv
    auto beta = (u_x * r_x + u_y * r_y + u_z * r_z) * inv;

    // gamma = dot(v, d) * inv
    auto gamma = (v_x * d_x + v_y * d_y + v_z * d_z) * inv;

    *resultT = t;
    *resultA = beta;
    *resultB = gamma;

    float4 _zero = make_float4(0);
    float4 _one = make_float4(1);

    int32_t res_b0 = cmpGEQ(beta, _zero);        // beta >= 0
    int32_t res_b1 = cmpLEQ(beta, _one);        // beta <= 1

    int32_t res_g0 = cmpGEQ(gamma, _zero);        // gamma >= 0
    int32_t res_g1 = cmpLEQ(gamma, _one);        // gamma <= 1

    int32_t res_bg1 = cmpLEQ(beta + gamma, _one);    // beta + gammma <= 1

    int32_t res_t0 = cmpGEQ(t, _zero);        // t >= 0

    int32_t ret = res_b0 & res_b1 & res_g0 & res_g1 & res_bg1 & res_t0;

    return ret;
}

AT_INLINE_RELEASE __device__ int32_t hit4AABBWith1Ray(
    aten::vec4* result,
    const aten::vec3& org,
    const aten::vec3& dir,
    const float4& bminx, const float4& bmaxx,
    const float4& bminy, const float4& bmaxy,
    const float4& bminz, const float4& bmaxz,
    float t_min, float t_max)
{
    float4 invdx = make_float4(1.0f / (dir.x + 1e-6f));
    float4 invdy = make_float4(1.0f / (dir.y + 1e-6f));
    float4 invdz = make_float4(1.0f / (dir.z + 1e-6f));

    float4 ox = make_float4(-org.x * invdx.x);
    float4 oy = make_float4(-org.y * invdy.x);
    float4 oz = make_float4(-org.z * invdz.x);

    // X
    auto fx = bmaxx * invdx + ox;
    auto nx = bminx * invdx + ox;

    // Y
    auto fy = bmaxy * invdy + oy;
    auto ny = bminy * invdy + oy;

    // Z
    auto fz = bmaxz * invdz + oz;
    auto nz = bminz * invdz + oz;

    auto tmaxX = aten::vmax(fx, nx);
    auto tminX = aten::vmin(fx, nx);

    auto tmaxY = aten::vmax(fy, ny);
    auto tminY = aten::vmin(fy, ny);

    auto tmaxZ = aten::vmax(fz, nz);
    auto tminZ = aten::vmin(fz, nz);

    auto t1 = aten::vmin(aten::vmin(tmaxX, tmaxY), aten::vmin(tmaxZ, make_float4(t_max)));
    auto t0 = aten::vmax(aten::vmax(tminX, tminY), aten::vmax(tminZ, make_float4(t_min)));

    union isHit {
        struct {
            uint8_t _0 : 1;
            uint8_t _1 : 1;
            uint8_t _2 : 1;
            uint8_t _3 : 1;
            uint8_t padding : 4;
        };
        uint8_t f;
    } hit;

    hit.f = 0;
    hit._0 = (t0.x <= t1.x);
    hit._1 = (t0.y <= t1.y);
    hit._2 = (t0.z <= t1.z);
    hit._3 = (t0.w <= t1.w);

    result->x = hit._0 ? t0.x : AT_MATH_INF;
    result->y = hit._1 ? t0.y : AT_MATH_INF;
    result->z = hit._2 ? t0.z : AT_MATH_INF;
    result->w = hit._3 ? t0.w : AT_MATH_INF;

    return hit.f;
}

inline __device__ float4 cross(float4 a, float4 b)
{
    return make_float4(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
        0);
}
