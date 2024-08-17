#pragma once

#include "camera/camera.h"
#include "math/vec3.h"
#include "sampler/sampler.h"

namespace aten {
    class ThinLensCamera : public Camera {
    public:
        ThinLensCamera() = default;
        virtual ~ThinLensCamera() = default;

    public:
        void init(
            int32_t width, int32_t height,
            vec3 lookfrom, vec3 lookat, vec3 vup,
            float imageSensorSize,
            float imageSensorToLensDistance,
            float lensToObjectplaneDistance,
            float lensRadius,
            float W_scale);

        virtual void update() override final;

        virtual CameraSampleResult sample(
            float s, float t,
            sampler* sampler) const override final;

        virtual float HitOnLens(
            const ray& r,
            vec3& pos_on_lens,
            vec3& pos_on_object_plane,
            vec3& pos_on_image_sensor,
            int32_t& x, int32_t& y) const override final;

        virtual float ConvertImageSensorPdfToScenePdf(
            float pdf_image,
            const vec3& hit_point,
            const vec3& hit_point_nml,
            const vec3& pos_on_image_sensor,
            const vec3& pos_on_lens,
            const vec3& pos_on_object_plane) const override final;

        virtual float GetSensitivity(
            const vec3& pos_on_image_sensor,
            const vec3& pos_on_lens) const override final;

        virtual float GetWdash(
            const vec3& hit_point,
            const vec3& hit_point_nml,
            const vec3& pos_on_image_sensor,
            const vec3& pos_on_lens,
            const vec3& pos_on_object_plane) const override final;

        virtual bool NeedRevert() const final
        {
            return true;
        }

        virtual bool IsPinhole() const final
        {
            return false;
        }

        virtual const vec3& GetPos() const override final
        {
            return m_imagesensor.center;
        }
        virtual const vec3& GetDir() const override final
        {
            return m_imagesensor.dir;
        }

        virtual aten::vec3& GetPos() override final
        {
            return m_imagesensor.center;
        }
        virtual aten::vec3& GetAt() override final
        {
            return m_at;
        }

        void RevertRayToPixelPos(
            const ray& ray,
            int32_t& px, int32_t& py) const override final;

        virtual float GetImageSensorWidth() const final
        {
            return m_imagesensor.width;
        }

        virtual float GetImageSensorHeight() const final
        {
            return m_imagesensor.height;
        }

    private:
        // 解像度.
        int32_t m_imageWidthPx;
        int32_t m_imageHeightPx;

        // イメージセンサ.
        struct ImageSensor {
            vec3 center;
            vec3 dir;
            vec3 up;
            vec3 u;
            vec3 v;
            vec3 lower_left;
            float width;
            float height;
        } m_imagesensor;

        // 物理的なピクセルサイズ.
        float m_pixelWidth;
        float m_pixelHeight;

        // レンズ.
        struct Lens {
            vec3 center;
            vec3 u;
            vec3 v;
            vec3 normal;
            float radius;
        } m_lens;

        struct ObjectPlane {
            vec3 center;
            vec3 u;
            vec3 v;
            vec3 normal;
            vec3 lower_left;
        } m_objectplane;

        float m_imageSensorToLensDistance;
        float m_lensToObjectplaneDistance;

        // イメージセンサの感度
        float m_W;

        vec3 m_at;    // 注視点.
        vec3 m_vup;
        float m_Wscale;
    };
}
