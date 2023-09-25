#include "geometry/sphere.h"

namespace AT_NAME
{
    static inline AT_DEVICE_API void getUV(real& u, real& v, const aten::vec3& p)
    {
        auto phi = aten::asin(p.y);
        auto theta = aten::atan(p.x / p.z);

        u = (theta + AT_MATH_PI_HALF) / AT_MATH_PI;
        v = (phi + AT_MATH_PI_HALF) / AT_MATH_PI;
    }

    bool sphere::hit(
        const aten::context& ctxt,
        const aten::ray& r,
        real t_min, real t_max,
        aten::Intersection& isect) const
    {
        bool isHit = hit(&m_param, r, t_min, t_max, &isect);

        if (isHit) {
            isect.objid = id();
            isect.mtrlid = mtrl_->id();
        }

        return isHit;
    }

    bool AT_DEVICE_API sphere::hit(
        const aten::ObjectParameter* param,
        const aten::ray& r,
        real t_min, real t_max,
        aten::Intersection* isect)
    {
        // NOTE
        // https://www.slideshare.net/h013/edupt-kaisetsu-22852235
        // p52 - p58

        const aten::vec3 p_o = param->sphere.center - r.org;
        const real b = dot(p_o, r.dir);

        // 判別式.
        const real D4 = b * b - dot(p_o, p_o) + param->sphere.radius * param->sphere.radius;

        if (D4 < real(0)) {
            return false;
        }

        const real sqrt_D4 = aten::sqrt(D4);
        const real t1 = b - sqrt_D4;
        const real t2 = b + sqrt_D4;

#if 0
        if (t1 > AT_MATH_EPSILON) {
            isect->t = t1;
        }
        else if (t2 > AT_MATH_EPSILON) {
            isect->t = t2;
        }
        else {
            return false;
        }
#elif 1
        // TODO
        // maxUlps の値によって、RandomSceneのreflactionがうまくいかないことがある...
        bool close = aten::isClose(aten::abs(b), sqrt_D4, 2500);

        if (t1 > AT_MATH_EPSILON && !close) {
            isect->t = t1;
        }
        else if (t2 > AT_MATH_EPSILON && !close) {
            isect->t = t2;
        }
        else {
            return false;
        }
#else
        if (t1 < 0 && t2 < 0) {
            return false;
        }
        else if (t1 > 0 && t2 > 0) {
            isect->t = aten::cmpMin(t1, t2);
        }
        else {
            isect->t = aten::cmpMax(t1, t2);
        }
#endif

        return true;
    }

    void sphere::evalHitResult(
        const aten::ObjectParameter* param,
        const aten::ray& r,
        aten::hitrecord* rec,
        const aten::Intersection* isect)
    {
        rec->p = r.org + isect->t * r.dir;
        rec->normal = (rec->p - param->sphere.center) / param->sphere.radius; // 正規化して法線を得る

        rec->mtrlid = isect->mtrlid;

        auto radius = param->sphere.radius;

        rec->area = 4 * AT_MATH_PI * radius * radius;

        getUV(rec->u, rec->v, rec->normal);
    }

    void sphere::sample_pos_and_normal(
        aten::SamplePosNormalPdfResult* result,
        const aten::ObjectParameter& param,
        const aten::mat4& mtx_L2W,
        aten::sampler* sampler)
    {
        auto r1 = sampler->nextSample();
        auto r2 = sampler->nextSample();

        auto r = param.sphere.radius;

        auto z = real(2) * r1 - real(1); // [0,1] -> [-1, 1]

        auto sin_theta = aten::sqrt(1 - z * z);
        auto phi = 2 * AT_MATH_PI * r2;

        auto x = aten::cos(phi) * sin_theta;
        auto y = aten::sin(phi) * sin_theta;

        aten::vec3 dir = aten::vec3(x, y, z);
        dir = normalize(dir);

        auto p = dir * (r + AT_MATH_EPSILON);

        result->pos = param.sphere.center + p;

        result->nml = normalize(result->pos - param.sphere.center);

        result->area = real(1);
        {
            auto tmp = param.sphere.center + aten::vec3(param.sphere.radius, 0, 0);

            auto center = mtx_L2W.apply(param.sphere.center);
            tmp = mtx_L2W.apply(tmp);

            auto radius = length(tmp - center);

            result->area = 4 * AT_MATH_PI * radius * radius;
        }
    }
}
