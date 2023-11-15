#pragma once

#include <memory>

#include "types.h"
#include "accelerator/bvh.h"
#include "math/mat4.h"
#include "geometry/PolygonObject.h"
#include "deformable/deformable.h"
#include "scene/host_scene_context.h"

namespace aten
{
    template <typename OBJ>
    class instance : public transformable {
        friend class TransformableFactory;

    public:
        instance(const std::shared_ptr<OBJ>& obj, context& ctxt)
            : transformable(ObjectType::Instance), m_obj(obj)
        {
            init_matrices(ctxt);
            m_param.object_id = m_obj->id();
            setBoundingBox(m_obj->getBoundingbox());
        }

        instance(const std::shared_ptr<OBJ>& obj, context& ctxt, const mat4& mtx_L2W)
            : instance(obj, ctxt)
        {
            init_matrices(ctxt);

            *m_mtx_L2W = mtx_L2W;

            *m_mtx_W2L = mtx_L2W;
            m_mtx_W2L->invert();

            setBoundingBox(getTransformedBoundingBox());
        }

        instance(
            const std::shared_ptr<OBJ>& obj,
            context& ctxt,
            const vec3& trans,
            const vec3& rot,
            const vec3& scale)
            : instance(obj, ctxt)
        {
            init_matrices(ctxt);

            m_trans = trans;
            m_rot = rot;
            m_scale= scale;

            updateMatrix();

            *m_mtx_W2L = *m_mtx_L2W;
            m_mtx_W2L->invert();

            setBoundingBox(getTransformedBoundingBox());
        }

        virtual ~instance() = default;

        virtual bool hit(
            const context& ctxt,
            const ray& r,
            real t_min, real t_max,
            Intersection& isect) const override final
        {
            vec3 org = r.org;
            vec3 dir = r.dir;

            // Transform world to local.
            org = m_mtx_W2L->apply(org);
            dir = m_mtx_W2L->applyXYZ(dir);

            ray transformdRay(org, dir);

            // Hit test in local coordinate.
            auto isHit = m_obj->hit(ctxt, transformdRay, t_min, t_max, isect);

            if (isHit) {
                // returnTo this instance's id.
                isect.objid = id();
            }

            return isHit;
        }

        virtual const hitable* getHasObject() const override final
        {
            return m_obj.get();
        }

        OBJ* getHasObjectAsRealType()
        {
            return m_obj.get();
        }

        virtual const hitable* getHasSecondObject() const override final
        {
            return m_lod.get();
        }

        virtual void getMatrices(
            aten::mat4& mtx_L2W,
            aten::mat4& mtx_W2L) const override final
        {
            mtx_L2W = *m_mtx_L2W;
            mtx_W2L = *m_mtx_W2L;
        }

        virtual aabb getTransformedBoundingBox() const override
        {
            return aabb::transform(m_obj->getBoundingbox(), *m_mtx_L2W);
        }

        virtual void render(
            aten::hitable::FuncPreDraw func,
            const context& ctxt,
            const aten::mat4& mtx_L2W,
            const aten::mat4& mtx_prev_L2W,
            int32_t parentId,
            uint32_t triOffset) override final
        {
            m_obj->render(func, ctxt, *m_mtx_L2W, m_mtx_prev_L2W, id(), triOffset);
        }

        virtual void drawAABB(
            aten::hitable::FuncDrawAABB func,
            const aten::mat4& mtx_L2W) override final
        {
            m_obj->drawAABB(func, *m_mtx_L2W);
        }

        virtual bool isDeformable() const override final
        {
            return m_obj->isDeformable();
        }

        virtual void update(bool isForcibly = false)
        {
            if (m_isDirty || isForcibly) {
                updateMatrix();
                setBoundingBox(getTransformedBoundingBox());
                onNotifyChanged();

                m_isDirty = false;
            }
        }

        void setLod(const std::shared_ptr<OBJ>& obj)
        {
            m_lod = obj;
        }

        vec3 getTrans()
        {
            return m_trans;
        }
        vec3 getRot()
        {
            return m_rot;
        }
        vec3 getScale()
        {
            return m_scale;
        }

        void setTrans(const vec3& trans)
        {
            m_isDirty = true;
            m_trans = trans;
        }
        void setRot(const vec3& rot)
        {
            m_isDirty = true;
            m_rot = rot;
        }
        void setScale(const vec3& scale)
        {
            m_isDirty = true;
            m_scale = scale;
        }

    private:
        void updateMatrix()
        {
            mat4 mtxTrans;
            mat4 mtxRotX;
            mat4 mtxRotY;
            mat4 mtxRotZ;
            mat4 mtxScale;

            mtxTrans.asTrans(m_trans);
            mtxRotX.asRotateByX(m_rot.x);
            mtxRotY.asRotateByX(m_rot.y);
            mtxRotZ.asRotateByX(m_rot.z);
            mtxScale.asScale(m_scale);

            // Keep previous L2W matrix.
            m_mtx_prev_L2W = *m_mtx_L2W;

            *m_mtx_L2W = mtxTrans * mtxRotX * mtxRotY * mtxRotZ * mtxScale;

            *m_mtx_W2L = *m_mtx_L2W;
            m_mtx_W2L->invert();
        }

        void init_matrices(context& ctxt)
        {
            std::tie(m_param.mtx_id, m_mtx_L2W) = ctxt.CreateMatrix();
            std::tie(std::ignore, m_mtx_W2L) = ctxt.CreateMatrix();
        }

    private:
        std::shared_ptr<OBJ> m_obj;
        std::shared_ptr<OBJ> m_lod;

        std::shared_ptr<mat4> m_mtx_L2W;
        std::shared_ptr<mat4> m_mtx_W2L;    // inverted.
        mat4 m_mtx_prev_L2W;

        vec3 m_trans;
        vec3 m_rot;
        vec3 m_scale{ aten::vec3(1, 1, 1) };

        bool m_isDirty{ false };
    };

    template<>
    inline instance<PolygonObject>::instance(const std::shared_ptr<PolygonObject>& obj, context& ctxt)
        : transformable(ObjectType::Instance), m_obj(obj)
    {
        m_param.object_id = m_obj->id();
        m_obj->build(ctxt);
        setBoundingBox(m_obj->getBoundingbox());
    }

    template<>
    inline instance<deformable>::instance(const std::shared_ptr<deformable>& obj, context& ctxt)
        : transformable(ObjectType::Instance), m_obj(obj)
    {
        m_param.object_id = m_obj->id();
        m_obj->build();
        setBoundingBox(m_obj->getBoundingbox());
    }
}
