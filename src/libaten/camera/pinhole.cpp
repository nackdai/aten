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

        result->pos_on_lens = s * param->u + t * param->v;
        result->pos_on_lens = result->pos_on_lens + param->center;

        result->r.dir = normalize(result->pos_on_lens - param->origin);

        result->nml_on_lens = param->dir;
        result->pos_on_image_sensor = param->origin;

        result->r.org = param->origin;

        result->pdf_on_lens = 1;
        result->pdf_on_image_sensor = 1;
    }

    void PinholeCamera::RevertRayToPixelPos(
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

    float PinholeCamera::ConvertImageSensorPdfToScenePdf(
        float pdf_image,    // Not used.
        const aten::vec3& hit_point,
        const aten::vec3& hit_point_nml,
        const aten::vec3& pos_on_image_sensor,
        const aten::vec3& pos_on_lens,
        const aten::vec3& pos_on_object_plane) const
    {
        float pdf = float(1) / (m_param.width * m_param.height);

        aten::vec3 v = hit_point - pos_on_lens;

        {
            aten::vec3 dir = normalize(v);
            const float cosTheta = dot(dir, m_param.dir);
            const float dist = m_param.dist / (cosTheta + float(0.0001));
            const float dist2 = dist * dist;
            pdf = pdf / (cosTheta / dist2);
        }

        {
            aten::vec3 dv = hit_point - pos_on_lens;
            const float dist2 = aten::squared_length(dv);
            dv = normalize(dv);
            const float c = dot(hit_point_nml, dv);

            pdf = pdf * aten::abs(c / dist2);
        }

        return pdf;
    }

    float PinholeCamera::GetWdash(
        const aten::vec3& hit_point,
        const aten::vec3& hit_point_nml,
        const aten::vec3& pos_on_image_sensor,
        const aten::vec3& pos_on_lens,
        const aten::vec3& pos_on_object_plane) const
    {
        const float W = float(1) / (m_param.width * m_param.height);

        aten::vec3 v = hit_point - pos_on_lens;
        const float dist = length(v);
        v = normalize(v);

        // imagesensor -> lens
        const float c0 = dot(v, m_param.dir);
        const float d0 = m_param.dist / c0;
        const float G0 = c0 / (d0 * d0);

        // hitpoint -> camera
        const float c1 = dot(normalize(hit_point_nml), -v);
        const float d1 = dist;
        const float G1 = c1 / (d1 * d1);

        float W_dash = W / G0 * G1;

        return W_dash;
    }

    float PinholeCamera::HitOnLens(
        const aten::ray& r,
        aten::vec3& pos_on_lens,
        aten::vec3& pos_on_object_plane,
        aten::vec3& pos_on_image_sensor,
        int32_t& x, int32_t& y) const
    {
        int32_t px;
        int32_t py;

        RevertRayToPixelPos(r, px, py);

        if ((px >= 0) && (px < m_param.width)
            && (py >= 0) && (py < m_param.height))
        {
            x = px;
            y = py;

            float u = (float)x / (float)m_param.width;
            float v = (float)y / (float)m_param.height;

            auto camsample = sample(u, v, nullptr);
            pos_on_lens = camsample.pos_on_lens;

            float lens_t = length(pos_on_lens - r.org);

            return lens_t;
        }

        return -AT_MATH_INF;
    }

    void PinholeCamera::FitBoundingBox(
        const aten::aabb& bounding_box,
        bool is_dir_to_curr_cam_param/*= false*/)
    {
        aten::vec3 cam_origin, lookat;
        aten::tie(cam_origin, lookat) = PinholeCamera::FitBoundingBox(m_param, bounding_box, is_dir_to_curr_cam_param);

        Initalize(
            cam_origin, lookat,
            aten::vec3(0, 1, 0),
            m_param.vfov,
            m_param.znear, m_param.zfar,
            m_param.width, m_param.height);
    }

    aten::tuple<aten::vec3, aten::vec3> PinholeCamera::FitBoundingBox(
        const aten::CameraParameter& param,
        const aten::aabb& bounding_box,
        bool is_dir_to_curr_cam_param/*= false*/)
    {
        const auto bbox_center = bounding_box.getCenter();

        const float theta = aten::Deg2Rad(param.vfov);
        const auto distance = bounding_box.ComputeDistanceToCoverBoundingSphere(theta);

        aten::vec3 cam_origin(bbox_center.x, bbox_center.y, bbox_center.z * 2);
        if (is_dir_to_curr_cam_param) {
            cam_origin = param.origin;
        }

        const auto dir = normalize(cam_origin - bbox_center);

        cam_origin = bbox_center + dir * distance;
        const auto lookat = bbox_center;

        return aten::make_tuple(cam_origin, lookat);
    }
}
