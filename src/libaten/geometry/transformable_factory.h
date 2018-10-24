#pragma once

#include "geometry/transformable.h"
#include "geometry/sphere.h"
#include "geometry/cube.h"
#include "geometry/object.h"
#include "scene/instance.h"
#include "scene/context.h"
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
        static sphere* createSphere(
            context& ctxt,
            const aten::vec3& center, 
            real radius, 
            material* mtrl)
        {
            sphere* ret = new sphere(center, radius, mtrl);
            AT_ASSERT(ret);

            ctxt.addTransformable(ret);

            return ret;
        }

        static cube* createCube(
            context& ctxt,
            const aten::vec3& center,
            real w, real h, real d,
            material* mtrl)
        {
            cube* ret = new cube(center, w, h, d, mtrl);
            AT_ASSERT(ret);

            ctxt.addTransformable(ret);

            return ret;
        }

        static object* createObject(context& ctxt)
        {
            object* ret = new object();
            AT_ASSERT(ret);

            ctxt.addTransformable(ret);

            return ret;
        }

        template<class T>
        static instance<T>* createInstance(
            context& ctxt,
            T* obj)
        {
            auto ret = new instance<T>(obj, ctxt);
            AT_ASSERT(ret);

            ctxt.addTransformable(ret);

            return ret;
        }

        template<class T>
        static instance<T>* createInstance(
            context& ctxt,
            T* obj,
            const mat4& mtxL2W)
        {
            auto ret = new instance<T>(obj, ctxt, mtxL2W);
            AT_ASSERT(ret);

            ctxt.addTransformable(ret);

            return ret;
        }

        template<class T>
        static instance<T>* createInstance(
            context& ctxt,
            T* obj,
            const vec3& trans,
            const vec3& rot,
            const vec3& scale)
        {
            auto ret = new instance<T>(obj, ctxt, trans, rot, scale);
            AT_ASSERT(ret);

            ctxt.addTransformable(ret);

            return ret;
        }

        static deformable* createDeformable(context& ctxt)
        {
            auto ret = new deformable();
            AT_ASSERT(ret);

            ctxt.addTransformable(ret);

            return ret;
        }
    };
}
