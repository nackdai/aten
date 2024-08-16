#include "camera/pinhole.h"
#include "math/aabb.h"

namespace AT_NAME {
    void PinholeCamera::init(
        const aten::vec3& origin,
        const aten::vec3& lookat,
        const aten::vec3& up,
        float vfov,    // vertical fov.
        int32_t width, int32_t height)
    {
        m_param = CreateCameraParam(
            origin, lookat, up,
            vfov,
            0.0F, 0.0F,
            width, height);
    }

    void PinholeCamera::Initalize(
        const aten::vec3& origin,
        const aten::vec3& lookat,
        const aten::vec3& up,
        float vfov,
        float z_near, float z_far,
        int32_t width, int32_t height)
    {
        m_param = CreateCameraParam(
            origin, lookat, up,
            vfov,
            z_near, z_far,
            width, height);
    }

    aten::CameraParameter PinholeCamera::CreateCameraParam(
        const aten::vec3& origin,
        const aten::vec3& lookat,
        const aten::vec3& up,
        float vfov,
        float z_near, float z_far,
        int32_t width, int32_t height)
    {
        aten::CameraParameter param;

        float theta = aten::Deg2Rad(vfov);

        param.aspect = width / (float)height;

        float half_height = aten::tan(theta / 2);
        float half_width = param.aspect * half_height;

        param.origin = origin;
        param.lookat = lookat;

        // カメラ座標ベクトル.
        param.dir = normalize(lookat - origin);
        param.right = normalize(cross(param.dir, up));
        param.up = cross(param.right, param.dir);

        param.center = origin + param.dir;

        // スクリーンのUVベクトル.
        param.u = half_width * param.right;
        param.v = half_height * param.up;

        param.dist = height / (float(2.0) * aten::tan(theta / 2));

        param.vfov = vfov;
        param.width = width;
        param.height = height;

        param.znear = std::min(z_near, z_far);
        param.zfar = std::max(z_near, z_far);

        return param;
    }

    void PinholeCamera::update()
    {
        init(
            m_param.origin,
            m_param.lookat,
            m_param.up,
            m_param.vfov,
            m_param.width,
            m_param.height);
    }

    CameraSampleResult PinholeCamera::sample(
        float s, float t,
        aten::sampler* sampler) const
    {
        CameraSampleResult result;
        sample(&result, &m_param, s, t);
        return result;
    }

    AT_HOST_DEVICE_API void PinholeCamera::sample(
        CameraSampleResult* result,
        const aten::CameraParameter* param,
        float s, float t)
    {
        // [0, 1] -> [-1, 1]
        s = float(2) * s - float(1);
        t = float(2) * t - float(1);

        result->posOnLens = s * param->u + t * param->v;
        result->posOnLens = result->posOnLens + param->center;

        result->r.dir = normalize(result->posOnLens - param->origin);

        result->nmlOnLens = param->dir;
        result->posOnImageSensor = param->origin;

        result->r.org = param->origin;

        result->pdfOnLens = 1;
        result->pdfOnImageSensor = 1;
    }

    void PinholeCamera::revertRayToPixelPos(
        const aten::ray& ray,
        int32_t& px, int32_t& py) const
    {
        // dir 方向へのスクリーン距離.
        //     /|
        //  x / |
        //   /  |
        //  / θ|
        // +----+
        //    d
        // cosθ = x / d => x = d / cosθ

        float c = dot(ray.dir, m_param.dir);
        float dist = m_param.dist / c;

        aten::vec3 screenPos = m_param.origin + ray.dir * dist - m_param.center;

        float u = dot(screenPos, m_param.right) + m_param.width * float(0.5);
        float v = dot(screenPos, m_param.up) + m_param.height * float(0.5);

        px = (int32_t)u;
        py = (int32_t)v;
    }

    float PinholeCamera::convertImageSensorPdfToScenePdf(
        float pdfImage,    // Not used.
        const aten::vec3& hitPoint,
        const aten::vec3& hitpointNml,
        const aten::vec3& posOnImageSensor,
        const aten::vec3& posOnLens,
        const aten::vec3& posOnObjectPlane) const
    {
        float pdf = float(1) / (m_param.width * m_param.height);

        aten::vec3 v = hitPoint - posOnLens;

        {
            aten::vec3 dir = normalize(v);
            const float cosTheta = dot(dir, m_param.dir);
            const float dist = m_param.dist / (cosTheta + float(0.0001));
            const float dist2 = dist * dist;
            pdf = pdf / (cosTheta / dist2);
        }

        {
            aten::vec3 dv = hitPoint - posOnLens;
            const float dist2 = aten::squared_length(dv);
            dv = normalize(dv);
            const float c = dot(hitpointNml, dv);

            pdf = pdf * aten::abs(c / dist2);
        }

        return pdf;
    }

    float PinholeCamera::getWdash(
        const aten::vec3& hitPoint,
        const aten::vec3& hitpointNml,
        const aten::vec3& posOnImageSensor,
        const aten::vec3& posOnLens,
        const aten::vec3& posOnObjectPlane) const
    {
        const float W = float(1) / (m_param.width * m_param.height);

        aten::vec3 v = hitPoint - posOnLens;
        const float dist = length(v);
        v = normalize(v);

        // imagesensor -> lens
        const float c0 = dot(v, m_param.dir);
        const float d0 = m_param.dist / c0;
        const float G0 = c0 / (d0 * d0);

        // hitpoint -> camera
        const float c1 = dot(normalize(hitpointNml), -v);
        const float d1 = dist;
        const float G1 = c1 / (d1 * d1);

        float W_dash = W / G0 * G1;

        return W_dash;
    }

    float PinholeCamera::hitOnLens(
        const aten::ray& r,
        aten::vec3& posOnLens,
        aten::vec3& posOnObjectPlane,
        aten::vec3& posOnImageSensor,
        int32_t& x, int32_t& y) const
    {
        int32_t px;
        int32_t py;

        revertRayToPixelPos(r, px, py);

        if ((px >= 0) && (px < m_param.width)
            && (py >= 0) && (py < m_param.height))
        {
            x = px;
            y = py;

            float u = (float)x / (float)m_param.width;
            float v = (float)y / (float)m_param.height;

            auto camsample = sample(u, v, nullptr);
            posOnLens = camsample.posOnLens;

            float lens_t = length(posOnLens - r.org);

            return lens_t;
        }

        return -AT_MATH_INF;
    }

    void PinholeCamera::FitBoundingBox(const aten::aabb& bounding_box)
    {
        aten::vec3 origin, lookat;
        aten::tie(origin, lookat) = PinholeCamera::FitBoundingBox(m_param, bounding_box);

        Initalize(
            origin, lookat,
            aten::vec3(0, 1, 0),
            m_param.vfov,
            m_param.znear, m_param.zfar,
            m_param.width, m_param.height);
    }

    aten::tuple<aten::vec3, aten::vec3> PinholeCamera::FitBoundingBox(
        const aten::CameraParameter& param,
        const aten::aabb& bounding_box)
    {
        const auto bbox_center = bounding_box.getCenter();

        const float theta = aten::Deg2Rad(param.vfov);
        const auto distance = bounding_box.ComputeDistanceToCoverBoundingSphere(theta);

        const aten::vec3 tmp_origin(bbox_center.x, bbox_center.y, bbox_center.z * 2);
        const auto dir = normalize(tmp_origin - bbox_center);

        const auto origin = bbox_center + dir * distance;
        const auto lookat = bbox_center;

        return aten::make_tuple(origin, lookat);
    }
}
