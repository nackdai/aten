#pragma once

#include "geometry/transformable.h"
#include "geometry/sphere.h"
#include "geometry/PolygonObject.h"
#include "geometry/instance.h"
#include "scene/host_scene_context.h"
#include "deformable/deformable.h"

namespace aten
{
    class TransformableFactory {
    private:
        TransformableFactory() = delete;
        ~TransformableFactory() = delete;

        TransformableFactory(const TransformableFactory& rhs) = delete;
        const TransformableFactory& operator=(const TransformableFactory& rhs) = delete;

    public:
        static std::shared_ptr<sphere> createSphere(
            context& ctxt,
            const aten::vec3& center,
            real radius,
            const std::shared_ptr<material>& mtrl)
        {
            auto ret = std::make_shared<AT_NAME::sphere>(center, radius, mtrl);
            AT_ASSERT(ret);

            ctxt.addTransformable(ret);

            return ret;
        }

        static std::shared_ptr<AT_NAME::PolygonObject> createObject(context& ctxt)
        {
            auto ret = std::make_shared<AT_NAME::PolygonObject>();
            AT_ASSERT(ret);

            ctxt.addTransformable(ret);

            return ret;
        }

        template <typename T>
        static std::shared_ptr<instance<T>> createInstance(
            context& ctxt,
            const std::shared_ptr<T>& obj)
        {
            auto ret = std::make_shared<instance<T>>(obj, ctxt);
            AT_ASSERT(ret);

            ctxt.addTransformable(ret);

            return ret;
        }

        template <typename T>
        static std::shared_ptr<instance<T>> createInstance(
            context& ctxt,
            const std::shared_ptr<T>& obj,
            const mat4& mtx_L2W)
        {
            auto ret = std::make_shared<instance<T>>(obj, ctxt, mtx_L2W);
            AT_ASSERT(ret);

            ctxt.addTransformable(ret);

            return ret;
        }

        template <typename T>
        static std::shared_ptr<instance<T>> createInstance(
            context& ctxt,
            const std::shared_ptr<T>& obj,
            const vec3& trans,
            const vec3& rot,
            const vec3& scale)
        {
            auto ret = std::make_shared<instance<T>>(obj, ctxt, trans, rot, scale);
            AT_ASSERT(ret);

            ctxt.addTransformable(ret);

            return ret;
        }

        static std::shared_ptr<deformable> createDeformable(context& ctxt)
        {
            auto ret = std::make_shared<deformable>();
            AT_ASSERT(ret);

            ctxt.addTransformable(ret);

            return ret;
        }
    };
}
