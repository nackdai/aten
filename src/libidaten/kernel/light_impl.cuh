#include "kernel/idatendefs.cuh"

AT_CUDA_INLINE __device__ void getTriangleSamplePosNormalArea(
    aten::SamplePosNormalPdfResult* result,
    idaten::Context* ctxt,
    const aten::ObjectParameter* shape,
    aten::sampler* sampler)
{
    // CPUコードと処理を合わせるためのダミー.
    int32_t dummy = sampler->nextSample();

    int32_t r = sampler->nextSample();
    int32_t basePrimIdx = aten::cmpMin(r * shape->triangle_num, shape->triangle_num - 1);

    int32_t primidx = basePrimIdx + shape->triangle_id;

    const aten::TriangleParameter* prim = &ctxt->prims[primidx];

    float4 _p0 = ctxt->GetPosition(prim->idx[0]);
    float4 _p1 = ctxt->GetPosition(prim->idx[1]);
    float4 _p2 = ctxt->GetPosition(prim->idx[2]);

    float4 _n0 = ctxt->GetNormal(prim->idx[0]);
    float4 _n1 = ctxt->GetNormal(prim->idx[1]);
    float4 _n2 = ctxt->GetNormal(prim->idx[2]);

    aten::vec3 p0 = aten::vec3(_p0.x, _p0.y, _p0.z);
    aten::vec3 p1 = aten::vec3(_p1.x, _p1.y, _p1.z);
    aten::vec3 p2 = aten::vec3(_p2.x, _p2.y, _p2.z);

    aten::vec3 n0 = aten::vec3(_n0.x, _n0.y, _n0.z);
    aten::vec3 n1 = aten::vec3(_n1.x, _n1.y, _n1.z);
    aten::vec3 n2 = aten::vec3(_n2.x, _n2.y, _n2.z);

#if 0
    // 0 <= a + b <= 1
    real a = sampler->nextSample();
    real b = sampler->nextSample();

    real d = a + b;

    if (d > 1) {
        a /= d;
        b /= d;
    }
#else
    real r0 = sampler->nextSample();
    real r1 = sampler->nextSample();

    real a = aten::sqrt(r0) * (real(1) - r1);
    real b = aten::sqrt(r0) * r1;
#endif

    // 重心座標系(barycentric coordinates).
    // v0基準.
    // p = (1 - a - b)*v0 + a*v1 + b*v2
    aten::vec3 p = (1 - a - b) * p0 + a * p1 + b * p2;

    aten::vec3 n = (1 - a - b) * n0 + a * n1 + b * n2;
    n = normalize(n);

    // 三角形の面積 = ２辺の外積の長さ / 2;
    auto e0 = p1 - p0;
    auto e1 = p2 - p0;
    auto area = real(0.5) * cross(e0, e1).length();

    result->pos = p;
    result->nml = n;
    result->area = area;

    result->a = a;
    result->b = b;

    result->triangle_id = primidx;

    real orignalLen = (p1 - p0).length();

    real scaledLen = 0;

    if (shape->mtx_id >= 0) {
        const auto& mtxL2W = ctxt->GetMatrix(shape->mtx_id * 2 + 0);

        {
            auto v0 = mtxL2W.apply(p0);
            auto v1 = mtxL2W.apply(p1);

            scaledLen = (v1 - v0).length();
        }
    }
    else {
        scaledLen = (p1 - p0).length();
    }

    real ratio = scaledLen / orignalLen;
    ratio = ratio * ratio;

    result->area = shape->area * ratio;
}

AT_CUDA_INLINE __device__  void sampleAreaLight(
    aten::LightSampleResult* result,
    idaten::Context* ctxt,
    const aten::LightParameter* light,
    const aten::vec3& org,
    aten::sampler* sampler)
{
    const aten::ObjectParameter* s = (light->objid >= 0
        ? &ctxt->GetObject(light->objid)
        : nullptr);

    aten::ray r;
    aten::hitrecord rec;
    aten::Intersection isect;

    if (sampler) {
        aten::SamplePosNormalPdfResult result;

        const aten::ObjectParameter* realShape = (s->object_id >= 0
            ? &ctxt->GetObject(s->object_id)
            : s);

        if (realShape->type == aten::ObjectType::Polygon) {
            getTriangleSamplePosNormalArea(&result, ctxt, realShape, sampler);
        }
        else if (realShape->type == aten::ObjectType::Sphere) {
            AT_NAME::sphere::sample_pos_and_normal(&result, *s, sampler);
        }
        else {
            // TODO
        }

        auto dir = result.pos - org;

        // NOTE
        // Just do hit test if ray hits the specified object directly.
        // We don't need to mind self-intersection.
        // Therefore, we don't need to add offset.
        r = aten::ray(org, dir);

        if (result.triangle_id >= 0) {
            isect.t = dir.length();

            isect.triangle_id = result.triangle_id;

            isect.a = result.a;
            isect.b = result.b;
        }
        else {
            // TODO
            // Only for sphere...
            AT_NAME::sphere::hit(s, r, AT_MATH_EPSILON, AT_MATH_INF, &isect);
        }
    }
    else {
        // TODO
        // Only for sphere...
        auto pos = s->sphere.center;
        auto dir = pos - org;

        // NOTE
        // Just do hit test if ray hits the specified object directly.
        // We don't need to mind self-intersection.
        // Therefore, we don't need to add offset.
        r = aten::ray(org, dir);

        AT_NAME::sphere::hit(s, r, AT_MATH_EPSILON, AT_MATH_INF, &isect);
    }

    evalHitResultForAreaLight(ctxt, s, r, &rec, &isect);

    AT_NAME::AreaLight::sample(&rec, light, org, sampler, result);
}

AT_CUDA_INLINE __device__  void sampleImageBasedLight(
    aten::LightSampleResult* result,
    idaten::Context* ctxt,
    const aten::LightParameter* light,
    const aten::vec3& org,
    const aten::vec3& normal,
    aten::sampler* sampler,
    int32_t lod)
{
    // TODO
    int32_t envmapidx = light->envmapidx;

#if 0
    real u = sampler->nextSample();
    real v = sampler->nextSample();

    auto pi2 = AT_MATH_PI * AT_MATH_PI;
    auto theta = AT_MATH_PI * v;

    // u, v -> direction.
    result->dir = AT_NAME::envmap::convertUVToDirection(u, v);
#else
    auto n = normal;
    auto t = aten::getOrthoVector(normal);
    auto b = normalize(cross(n, t));

    real r1 = sampler->nextSample();
    real r2 = sampler->nextSample();

    real sinpsi = aten::sin(2 * AT_MATH_PI * r1);
    real cospsi = aten::cos(2 * AT_MATH_PI * r1);
    real costheta = aten::pow(1 - r2, 0.5);
    real sintheta = aten::sqrt(1 - costheta * costheta);

    // returnTo the result
    result->dir = normalize(t * sintheta * cospsi + b * sintheta * sinpsi + n * costheta);

    auto uv = AT_NAME::envmap::convertDirectionToUV(result->dir);

    auto u = uv.x;
    auto v = uv.y;
#endif

    // TODO
    // シーンのAABBを覆う球上に配置されるようにするべき.
    result->pos = org + real(100000) * result->dir;

    result->pdf = dot(normal, result->dir) / AT_MATH_PI;

    //auto le = tex2D<float4>(ctxt->textures[envmapidx], u, v);
    auto le = tex2DLod<float4>(ctxt->textures[envmapidx], u, v, lod);

    result->le = aten::vec3(le.x, le.y, le.z);
    result->finalColor = result->le;
}

AT_CUDA_INLINE __device__ void sampleLight(
    aten::LightSampleResult* result,
    idaten::Context* ctxt,
    const aten::LightParameter* light,
    const aten::vec3& org,
    const aten::vec3& normal,
    aten::sampler* sampler,
    int32_t lod/*= 0*/)
{
    switch (light->type) {
    case aten::LightType::Area:
        sampleAreaLight(result, ctxt, light, org, sampler);
        break;
    case aten::LightType::IBL:
        sampleImageBasedLight(result, ctxt, light, org, normal, sampler, lod);
        break;
    case aten::LightType::Direction:
        AT_NAME::DirectionalLight::sample(light, org, sampler, result);
        break;
    case aten::LightType::Point:
        AT_NAME::PointLight::sample(light, org, sampler, result);
        break;
    case aten::LightType::Spot:
        AT_NAME::SpotLight::sample(light, org, sampler, result);
        break;
    }
}
