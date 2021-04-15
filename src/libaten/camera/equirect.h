#pragma once

#include "camera/camera.h"
#include "math/ray.h"
#include "math/vec3.h"
#include "math/math.h"

namespace aten {
    class EquirectCamera : public camera {
    public:
        EquirectCamera() = default;
        virtual ~EquirectCamera() = default;

        void init(
            vec3 origin, vec3 lookat, vec3 up,
            uint32_t width, uint32_t height)
        {
            auto aspect = width / (real)height;
            AT_ASSERT(aspect == 2);

            m_origin = origin;

            // カメラ座標ベクトル.
            m_dir = normalize(lookat - origin);
            m_right = normalize(cross(m_dir, up));
            m_up = cross(m_right, m_dir);

            // 値を保持.
            m_at = lookat;
            m_width = (real)width;
            m_height = (real)height;
        }

        virtual void update() override final
        {
            init(
                m_origin, m_at, m_up,
                (uint32_t)m_width, (uint32_t)m_height);
        }

        virtual CameraSampleResult sample(
            real s, real t,
            sampler* sampler) const override final;

        virtual const vec3& getPos() const override final
        {
            return m_origin;
        }
        virtual const vec3& getDir() const override final
        {
            return m_dir;
        }

        virtual aten::vec3& getPos() override final
        {
            return m_origin;
        }
        virtual aten::vec3& getAt() override final
        {
            return m_at;
        }

        void revertRayToPixelPos(
            const ray& ray,
            int& px, int& py) const override final
        {
            // Not supported...
            AT_ASSERT(false);
        }

    private:
        vec3 m_origin;

        vec3 m_dir;
        vec3 m_right;
        vec3 m_up;

        vec3 m_at;
        real m_width;
        real m_height;
    };
}
