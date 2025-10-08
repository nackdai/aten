#pragma once

#include "camera/camera.h"
#include "math/ray.h"
#include "math/vec3.h"
#include "math/math.h"

namespace aten {
    class EquirectCamera : public Camera {
    public:
        EquirectCamera() = default;
        virtual ~EquirectCamera() = default;

        void init(
            vec3 origin, vec3 lookat, vec3 up,
            int32_t width, int32_t height)
        {
            auto aspect = width / (float)height;
            AT_ASSERT(aspect == 2);

            m_origin = origin;

            // カメラ座標ベクトル.
            m_dir = normalize(lookat - origin);
            right_ = normalize(cross(m_dir, up));
            m_up = cross(right_, m_dir);

            // 値を保持.
            m_at = lookat;
            width_ = (float)width;
            height_ = (float)height;
        }

        virtual void update() override final
        {
            init(
                m_origin, m_at, m_up,
                (uint32_t)width_, (uint32_t)height_);
        }

        virtual CameraSampleResult sample(
            float s, float t,
            sampler* sampler) const override final;

        virtual const vec3& GetPos() const override final
        {
            return m_origin;
        }
        virtual const vec3& GetDir() const override final
        {
            return m_dir;
        }

        virtual aten::vec3& GetPos() override final
        {
            return m_origin;
        }
        virtual aten::vec3& GetAt() override final
        {
            return m_at;
        }

        void RevertRayToPixelPos(
            const ray& ray,
            int32_t& px, int32_t& py) const override final
        {
            // Not supported...
            AT_ASSERT(false);
        }

    private:
        vec3 m_origin;

        vec3 m_dir;
        vec3 right_;
        vec3 m_up;

        vec3 m_at;
        float width_;
        float height_;
    };
}
