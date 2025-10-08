#pragma once

#include "camera/camera.h"
#include "math/ray.h"
#include "math/vec3.h"
#include "math/math.h"
#include "misc/tuple.h"

namespace aten {
    class aabb;
}

namespace AT_NAME
{
    /**
     * @brief Pinhole camera.
     */
    class PinholeCamera : public Camera
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
        virtual const aten::vec3 &GetPos() const override final
        {
            return param_.origin;
        }

        /**
         * @brief Return camera's direction.
         */
        virtual const aten::vec3 &GetDir() const override final
        {
            return param_.dir;
        }

        /**
         * @brief Return camera's origin.
         */
        virtual aten::vec3 &GetPos() override final
        {
            return param_.origin;
        }

        /**
         * @brief Return camera's point of gaze
         */
        virtual aten::vec3 &GetAt() override final
        {
            return param_.lookat;
        }

        virtual const aten::CameraParameter &param() const override final
        {
            return param_;
        }

        virtual float ComputePixelWidthAtDistance(float distance_from_camera) const override
        {
            return Camera::ComputePixelWidthAtDistance(param_, distance_from_camera);
        }

        void RevertRayToPixelPos(
            const aten::ray &ray,
            int32_t &px, int32_t &py) const override final;

        virtual float ConvertImageSensorPdfToScenePdf(
            float pdf_image, // Not used.
            const aten::vec3 &hit_point,
            const aten::vec3 &hit_point_nml,
            const aten::vec3 &pos_on_image_sensor,
            const aten::vec3 &pos_on_lens,
            const aten::vec3 &pos_on_object_plane) const override final;

        virtual float GetWdash(
            const aten::vec3 &hit_point,
            const aten::vec3 &hit_point_nml,
            const aten::vec3 &pos_on_image_sensor,
            const aten::vec3 &pos_on_lens,
            const aten::vec3 &pos_on_object_plane) const override final;

        virtual float HitOnLens(
            const aten::ray &r,
            aten::vec3 &pos_on_lens,
            aten::vec3 &pos_on_object_plane,
            aten::vec3 &pos_on_image_sensor,
            int32_t &x, int32_t &y) const override final;

        void FitBoundingBox(
            const aten::aabb& bounding_box,
            bool will_use_curr_camera_origin = false);

        static aten::tuple<aten::vec3, aten::vec3> FitBoundingBox(
            const aten::CameraParameter& param,
            const aten::aabb& bounding_box,
            bool will_use_curr_camera_origin = false);

    private:
        aten::CameraParameter param_;
    };
}
