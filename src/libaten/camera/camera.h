#pragma once

#include "types.h"
#include "math/ray.h"
#include "math/mat4.h"
#include "sampler/sampler.h"

namespace aten {
    // TODO
    // Only for Pinhole Camera...

    /**
     * @brief Description of camera.
     */
    struct CameraParameter {
        vec3 origin;        ///< Camera origin.

        float aspect{ 0 };   ///< Aspect of screen size.
        vec3 center;        ///< origin + dir.

        vec3 u;             ///< Axis (right) of screen.
        vec3 v;             ///< Axis (up) of screen.

        vec3 dir;           ///< Camera direction vector (z).
        vec3 right;         ///< Camera right vector (x).
        vec3 up;            ///< Camera up vector(y).

        float dist{ 0 };     ///< Distance to the clip plane.
        float vfov{ 0 };     ///< Vertical Field of View.
        int32_t width{ 0 };     ///< Screen width.
        int32_t height{ 0 };    ///< Screen height.

        float znear{ 0 };    ///< Z Near plane.
        float zfar{ 0 };     ///< Z Far plane.
    };
};

namespace AT_NAME {
    /**
     * @brief Result to sample camera.
     */
    struct CameraSampleResult {
        aten::ray r;                        ///< Ray from the position on the image sensor to the position on the lens.
        aten::vec3 posOnImageSensor;        ///< Position on the image sensor.
        aten::vec3 posOnLens;               ///< Position on the lens.
        aten::vec3 nmlOnLens;               ///< Normal at the position on the lens.
        aten::vec3 posOnObjectplane;        ///< Position on the obhect plane.
        float pdfOnImageSensor{ float(1) };   ///< PDF to sample the image sensor.
        float pdfOnLens{ float(1) };          ///< PDF to sample the image lens.
    };

    /**
     * @brief Interface for camera.
     */
    class camera {
    public:
        camera() = default;
        virtual ~camera() = default;

        /**
         * @brief Update transformed camera parmaters.
         */
        virtual void update() = 0;

        /**
         * @brief Sample camera.
         */
        virtual CameraSampleResult sample(
            float s, float t,
            aten::sampler* sampler) const = 0;

        virtual float convertImageSensorPdfToScenePdf(
            float pdfImage,
            const aten::vec3& hitPoint,
            const aten::vec3& hitpointNml,
            const aten::vec3& posOnImageSensor,
            const aten::vec3& posOnLens,
            const aten::vec3& posOnObjectPlane) const
        {
            return float(1);
        }

        virtual float getSensitivity(
            const aten::vec3& posOnImagesensor,
            const aten::vec3& posOnLens) const
        {
            return float(1);
        }

        virtual float getWdash(
            const aten::vec3& hitPoint,
            const aten::vec3& hitpointNml,
            const aten::vec3& posOnImageSensor,
            const aten::vec3& posOnLens,
            const aten::vec3& posOnObjectPlane) const
        {
            return float(1);
        }

        virtual float hitOnLens(
            const aten::ray& r,
            aten::vec3& posOnLens,
            aten::vec3& posOnObjectPlane,
            aten::vec3& posOnImageSensor,
            int32_t& x, int32_t& y) const
        {
            return -AT_MATH_INF;
        }

        /**
         * @brief Return whether the rendering result with the camera needs to revert.
         */
        virtual bool needRevert() const
        {
            return false;
        }

        /**
         * @brief Return whether the camera is pinhole camera.
         */
        virtual bool isPinhole() const
        {
            return true;
        }

        /**
         * @brief Return camera's origin.
         */
        virtual const aten::vec3& getPos() const = 0;

        /**
         * @brief Return camera's point of gaze
         */
        virtual const aten::vec3& getDir() const = 0;

        /**
         * @brief Return camera's origin.
         */
        virtual aten::vec3& getPos() = 0;

        /**
         * @brief Return camera's point of gaze
         */
        virtual aten::vec3& getAt() = 0;

        /**
         * @brief Revert a ray to screen position.
         */
        virtual void revertRayToPixelPos(
            const aten::ray& ray,
            int32_t& px, int32_t& py) const = 0;

        virtual float getImageSensorWidth() const
        {
            return float(1);
        }

        virtual float getImageSensorHeight() const
        {
            return float(1);
        }

        virtual const aten::CameraParameter& param() const
        {
            AT_ASSERT(false);
            static const aten::CameraParameter tmp{};
            return tmp;
        }

        virtual float computePixelWidthAtDistance(float distanceFromCamera) const
        {
            AT_ASSERT(false);
            return float(0);
        }

        static AT_HOST_DEVICE_API float computePixelWidthAtDistance(
            const aten::CameraParameter& param,
            float distanceFromCamera)
        {
            AT_ASSERT(distanceFromCamera > float(0));
            distanceFromCamera = aten::abs(distanceFromCamera);

            // Compute horizontal FoV.
            auto hfov = param.vfov * param.height / float(param.width);
            hfov = aten::Deg2Rad(hfov);

            auto half_width = aten::tan(hfov / 2) * distanceFromCamera;
            auto width = half_width * 2;
            auto pixel_width = width / float(param.width);
            return pixel_width;
        }

        static void ComputeCameraMatrices(
            const aten::CameraParameter& param,
            aten::mat4& mtx_W2V,
            aten::mat4& mtx_V2C)
        {
            mtx_W2V.lookat(
                param.origin,
                param.center,
                param.up);

            mtx_V2C.perspective(
                param.znear,
                param.zfar,
                param.vfov,
                param.aspect);
        }

        static float ComputeScreenDistance(
            const aten::CameraParameter& param,
            const int32_t height)
        {
            return height / (2.0f * aten::tan(0.5f * param.vfov));
        }
    };
}
