#pragma once

#include "camera/camera.h"
#include "math/ray.h"
#include "math/vec3.h"
#include "math/math.h"

namespace aten {
    class aabb;
}

namespace AT_NAME
{
    /**
     * @brief Pinhole camera.
     */
    class PinholeCamera : public camera
    {
    public:
        PinholeCamera() = default;
        virtual ~PinholeCamera() = default;

        /**
         * @brief Initialize the camera.
         */
        void init(
            const aten::vec3 &origin,
            const aten::vec3 &lookat,
            const aten::vec3 &up,
            float vfov, // vertical fov.
            int32_t width, int32_t height);

        void Initalize(
            const aten::vec3& origin,
            const aten::vec3& lookat,
            const aten::vec3& up,
            float vfov,
            float z_near, float z_far,
            int32_t width, int32_t height);

        static aten::CameraParameter CreateCameraParam(
            const aten::vec3& origin,
            const aten::vec3& lookat,
            const aten::vec3& up,
            float vfov,
            float z_near, float z_far,
            int32_t width, int32_t height);

        /**
         * @brief Update transformed camera parmaters.
         */
        virtual void update() override final;

        /**
         * @brief Sample camera.
         */
        virtual CameraSampleResult sample(
            float s, float t,
            aten::sampler *sampler) const override final;

        /**
         * @brief Sample camera.
         */
        static AT_HOST_DEVICE_API void sample(
            CameraSampleResult *result,
            const aten::CameraParameter *param,
            float s, float t);

        /**
         * @brief Return camera's origin.
         */
        virtual const aten::vec3 &getPos() const override final
        {
            return m_param.origin;
        }

        /**
         * @brief Return camera's direction.
         */
        virtual const aten::vec3 &getDir() const override final
        {
            return m_param.dir;
        }

        /**
         * @brief Return camera's origin.
         */
        virtual aten::vec3 &getPos() override final
        {
            return m_param.origin;
        }

        /**
         * @brief Return camera's point of gaze
         */
        virtual aten::vec3 &getAt() override final
        {
            return m_at;
        }

        virtual const aten::CameraParameter &param() const override final
        {
            return m_param;
        }

        virtual float computePixelWidthAtDistance(float distanceFromCamera) const override
        {
            return camera::computePixelWidthAtDistance(m_param, distanceFromCamera);
        }

        void revertRayToPixelPos(
            const aten::ray &ray,
            int32_t &px, int32_t &py) const override final;

        virtual float convertImageSensorPdfToScenePdf(
            float pdfImage, // Not used.
            const aten::vec3 &hitPoint,
            const aten::vec3 &hitpointNml,
            const aten::vec3 &posOnImageSensor,
            const aten::vec3 &posOnLens,
            const aten::vec3 &posOnObjectPlane) const override final;

        virtual float getWdash(
            const aten::vec3 &hitPoint,
            const aten::vec3 &hitpointNml,
            const aten::vec3 &posOnImageSensor,
            const aten::vec3 &posOnLens,
            const aten::vec3 &posOnObjectPlane) const override final;

        virtual float hitOnLens(
            const aten::ray &r,
            aten::vec3 &posOnLens,
            aten::vec3 &posOnObjectPlane,
            aten::vec3 &posOnImageSensor,
            int32_t &x, int32_t &y) const override final;

        void FitBoundingBox(const aten::aabb& bounding_box);

        static aten::vec3 FitBoundingBox(
            const aten::CameraParameter& param,
            const aten::aabb& bounding_box);

    private:
        aten::CameraParameter m_param;
        aten::vec3 m_at;
    };
}
