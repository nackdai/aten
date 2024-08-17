#include "camera/thinlens.h"

namespace aten {
    // NOTE
    // http://kagamin.net/hole/edubpt/edubpt_v100.pdf

    static float plane_intersection(const vec3& normal, const vec3& pos, const ray& ray)
    {
        /*
            レイの方程式 : x = p + t v
            平面の方程式 : (p - p0)・n = 0
            p  : レイの出る点
            v  : レイの方向
            p0 : 平面上の１点
            n  : 法線ベクトル

            連立した結果、レイの進んだ距離tは

            (p0 - p)・n
            t = ------------
            v・n

            v・n=0 の時は、レイと平面が平行に進んでいるので、交わらない.
            t<0の時は、視線の後ろ側に交点があるので、やはり交わることがない.
        */

        float vn = dot(ray.dir, normal);

        if (fabs(vn) > AT_MATH_EPSILON) {
            float xpn = dot(pos - ray.org, normal);
            float t = xpn / vn;
            return t;
        }

        return AT_MATH_INF;
    }

    // NOTE
    // xI : イメージセンサー上の点.
    // x0 : レンズ上の点.
    // xV : オブジェクトプレーン上の点.
    // x1 : シーン内の点.

    void ThinLensCamera::init(
        int32_t width, int32_t height,
        vec3 lookfrom, vec3 lookat, vec3 vup,
        float imageSensorSize,
        float imageSensorToLensDistance,
        float lensToObjectplaneDistance,    // focus length
        float lensRadius,
        float W_scale)
    {
        // 値を保持.
        m_at = lookat;
        m_vup = vup;
        m_Wscale = W_scale;

        m_imageWidthPx = width;
        m_imageHeightPx = height;

        m_imageSensorToLensDistance = imageSensorToLensDistance;
        m_lensToObjectplaneDistance = lensToObjectplaneDistance;

        // イメージセンサのサイズ.
        m_imagesensor.width = imageSensorSize * m_imageWidthPx / m_imageHeightPx;
        m_imagesensor.height = imageSensorSize;

        // １ピクセルの物理サイズ.
        m_pixelWidth = m_imagesensor.width / m_imageWidthPx;
        m_pixelHeight = m_imagesensor.height / m_imageHeightPx;

        m_imagesensor.center = lookfrom;
        m_imagesensor.dir = normalize(lookat - lookfrom);
        m_imagesensor.u = normalize(cross(m_imagesensor.dir, vup)) * m_imagesensor.width;
        m_imagesensor.v = normalize(cross(m_imagesensor.u, m_imagesensor.dir)) * m_imagesensor.height;

        m_imagesensor.lower_left = m_imagesensor.center - 0.5f * m_imagesensor.u - 0.5f * m_imagesensor.v;

        // オブジェクトプレーンはイメージセンサーと平行なので、イメージセンサーと同じ方向を使えばいい.
        m_objectplane.center = m_imagesensor.center + (m_imageSensorToLensDistance + m_lensToObjectplaneDistance) * m_imagesensor.dir;
        m_objectplane.normal = m_imagesensor.dir;
        m_objectplane.u = m_imagesensor.u;
        m_objectplane.v = m_imagesensor.v;

        m_objectplane.lower_left = m_objectplane.center - 0.5f * m_objectplane.u - 0.5f * m_objectplane.v;

        // レンズはイメージセンサーと平行なので、イメージセンサーと同じ方向を使えばいい.
        m_lens.center = m_imagesensor.center + m_imageSensorToLensDistance * m_imagesensor.dir;
        m_lens.u = lensRadius * normalize(m_imagesensor.u);
        m_lens.v = lensRadius * normalize(m_imagesensor.v);
        m_lens.normal = m_imagesensor.dir;
        m_lens.radius = lensRadius;

        // W(xI <- x0) センサのセンシティビティは簡単のため定数にしておく.
        m_W = W_scale / (m_pixelWidth * m_pixelHeight);
    }

    void ThinLensCamera::update()
    {
        init(
            m_imageWidthPx, m_imageHeightPx,
            m_imagesensor.center, m_at, m_vup,
            m_imagesensor.height,    // image sensor size.
            m_imageSensorToLensDistance,
            m_lensToObjectplaneDistance,
            m_lens.radius,
            m_Wscale);
    }

    CameraSampleResult ThinLensCamera::sample(
        float s, float t,
        sampler* sampler) const
    {
        AT_ASSERT(sampler);

        // [0, 1] -> [-0.5, 0.5]
        s -= 0.5;
        t -= 0.5;

        CameraSampleResult result;

        // イメージセンサ上の座標、[-0,5, 0.5]の範囲（0,0)がイメージセンサ中央を示す.
        result.pos_on_image_sensor = m_imagesensor.center + s * m_imagesensor.u + t * m_imagesensor.v;

        // オブジェクトプレーン上の座標計算
        // オブジェクトプレーンのサイズは、レンズの公式の倍率計算（m=b/a）
        float ratio = m_lensToObjectplaneDistance / m_imageSensorToLensDistance;

        // センサーとオブジェクトプレーンの向きは反対になるので、マイナスする?
        float u_on_objectplane = -ratio * s;
        float v_on_objectplane = -ratio * t;
        result.pos_on_object_plane = m_objectplane.center + u_on_objectplane * m_objectplane.u + v_on_objectplane * m_objectplane.v;

        // NOTE
        // lens.u、lens.v に lens.radius が含まれているので、レンズの半径について考慮する必要がない.
        float r0 = sqrt(sampler->nextSample());
        float r1 = sampler->nextSample() * float(2.0) * AT_MATH_PI;
        float u = r0 * cos(r1);
        float v = r0 * sin(r1);
        result.pos_on_lens = m_lens.center + u * m_lens.u + v * m_lens.v;
        result.nml_on_lens = m_lens.normal;

        // ピクセル内の一点をサンプリングする確率密度関数（面積測度）
        result.pdf_on_image_sensor = float(1.0) / (m_pixelWidth * m_pixelHeight);

        // レンズ上の一点をサンプリングする確率密度関数（面積測度）
        result.pdf_on_lens = float(1.0) / (AT_MATH_PI * m_lens.radius * m_lens.radius);

        result.r = ray(
            result.pos_on_lens,
            normalize(result.pos_on_object_plane - result.pos_on_lens));

        return result;
    }

    float ThinLensCamera::HitOnLens(
        const ray& r,
        vec3& pos_on_lens,
        vec3& pos_on_object_plane,
        vec3& pos_on_image_sensor,
        int32_t& x, int32_t& y) const
    {
        // レンズと判定
        auto lens_t = plane_intersection(m_lens.normal, m_lens.center, r);

        if (AT_MATH_EPSILON < lens_t)
        {
            pos_on_lens = r.org + lens_t * r.dir;
            auto l = length(pos_on_lens - m_lens.center);
            auto d = dot(m_lens.normal, r.dir);

            if (l < m_lens.radius && d <= 0.0)
            {
                auto objplane_t = plane_intersection(m_objectplane.normal, m_objectplane.center, r);

                pos_on_object_plane = r.org + objplane_t * r.dir;

                auto u_on_objectplane = dot(pos_on_object_plane - m_objectplane.center, normalize(m_objectplane.u)) / length(m_objectplane.u);
                auto v_on_objectplane = dot(pos_on_object_plane - m_objectplane.center, normalize(m_objectplane.v)) / length(m_objectplane.v);

                auto ratio = m_imageSensorToLensDistance / m_lensToObjectplaneDistance;
                auto u = -ratio * u_on_objectplane;
                auto v = -ratio * v_on_objectplane;

                pos_on_image_sensor = m_imagesensor.center + u * m_imagesensor.u + v * m_imagesensor.v;

                if (-0.5f <= u && u < 0.5
                    && -0.5f <= v && v < 0.5)
                {
                    x = (int32_t)((u + 0.5f) * m_imageWidthPx);
                    y = (int32_t)((v + 0.5f) * m_imageHeightPx);

                    return lens_t;
                }
            }
        }

        return -AT_MATH_INF;
    }

    // イメージセンサ上のサンプリング確率密度（イメージセンサの面積測度に関する確率密度）をシーン上のサンプリング確率密度（面積測度に関する確率密度）に変換する.
    float ThinLensCamera::ConvertImageSensorPdfToScenePdf(
        float pdf_image,
        const vec3& hit_point,
        const vec3& hit_point_nml,
        const vec3& pos_on_image_sensor,
        const vec3& pos_on_lens,
        const vec3& pos_on_object_plane) const
    {
        // NOTE
        // http://rayspace.xyz/CG/contents/DoF.html

        // NOTE
        // http://kagamin.net/hole/edubpt/edubpt_v100.pdf
        // イメージセンサ上の確立密度を面積測度に変換.
        // p111
        // Pa = (B/A)^2 * (r''/r)^2 * (cosΘ'/cosΘ'') * Pij

        const vec3& x0 = pos_on_lens;
        const vec3& xI = pos_on_image_sensor;
        const vec3& xV = pos_on_object_plane;
        const vec3& x1 = hit_point;

        const vec3 x0_xV = xV - x0;
        const vec3 x0_x1 = x1 - x0;

        // NOTE
        // B = imagesensor_to_lens_distance
        // A = lens_to_objectplane(focus distance)

        // (B/A)^2
        const float ba = m_imageSensorToLensDistance / m_lensToObjectplaneDistance;
        const float ba2 = ba * ba;

        // (r''/r')^2
        auto r2 = (squared_length(x0_xV) / squared_length(x0_x1));

        // (cosΘ'/cosΘ'')
        auto c2 = (dot(normalize(-x0_x1), hit_point_nml) / dot(normalize(x0_x1), normalize(m_imagesensor.dir)));

        // イメージセンサの面積測度に関する確率密度.
        //auto pdf_on_image_sensor = 1.0 / (m_imagesensor.width * m_imagesensor.height);

        auto pdf = pdf_image * ba2 * r2 * c2;

        return pdf;
    }

    float ThinLensCamera::GetSensitivity(
        const vec3& pos_on_image_sensor,
        const vec3& pos_on_lens) const
    {
        // NOTE
        // http://kagamin.net/hole/edubpt/edubpt_v100.pdf
        // p54 - p62

        const vec3 x0_xI = pos_on_image_sensor - pos_on_lens;

        // センサ上の点の半球積分を、レンズ上の点の積分に変数変換した時に導入される係数
        // (cosΘ)^2/r^2
        const float cos = dot(normalize(-m_imagesensor.dir), normalize(x0_xI));
        const float len2 = squared_length(x0_xI);

        const float G = (cos * cos) / len2;

        float ret = m_W * G;

        return ret;
    }

    float ThinLensCamera::GetWdash(
        const vec3& hit_point,
        const vec3& hit_point_nml,
        const vec3& pos_on_image_sensor,
        const vec3& pos_on_lens,
        const vec3& pos_on_object_plane) const
    {
        // 幾何的な係数計算 + センサーセンシティビティの項を計算する.
        // x1 -> x0への放射輝度が最終的にイメージセンサに与える寄与度.

        const vec3& x0 = pos_on_lens;
        const vec3& xI = pos_on_image_sensor;
        const vec3& xV = pos_on_object_plane;
        const vec3& x1 = hit_point;

        const vec3 x0_xI = xI - x0;    // lens to image sensor.
        const vec3 x0_xV = xV - x0;    // lens to object plane.
        const vec3 x0_x1 = x1 - x0;    // lens to hit point.

        // NOTE
        // B = imagesensor_to_lens_distance
        // A = lens_to_objectplane(focus distance)

#if 1
        // B/A
        const float ba2 = squared_length(x0_xV) / squared_length(x0_xI);

        const float r = m_imageSensorToLensDistance / m_lensToObjectplaneDistance;
        const float r2 = r * r;

        const float c = dot(normalize(x0_xI), normalize(-m_imagesensor.dir)) / dot(normalize(x0_x1), normalize(m_imagesensor.dir));
        const float c2 = c * c;

        float W_dash = m_W * ba2 * r2 * c2;
#else
        // B/A
        const float ba = m_imageSensorToLensDistance / m_lensToObjectplaneDistance;
        const float ba2 = ba * ba;

        const float d0 = x0_xI.length();
        const float c0 = dot(normalize(x0_xI), normalize(-m_imagesensor.dir));
        const float G0 = c0 * c0 / (d0 * d0);

        const float d1 = x0_xV.length();
        const float c1 = dot(normalize(x0_x1), normalize(m_imagesensor.dir));
        const float G1 = c1 * c1 / (d1 * d1);

        float W_dash = m_W * ba2 / G0 * G1;
#endif

        return W_dash;
    }

    void ThinLensCamera::RevertRayToPixelPos(
        const ray& ray,
        int32_t& px, int32_t& py) const
    {
        vec3 pos_on_lens;
        vec3 pos_on_object_plane;
        vec3 pos_on_image_sensor;

        HitOnLens(
            ray,
            pos_on_lens,
            pos_on_object_plane,
            pos_on_image_sensor,
            px, py);
    }
}
