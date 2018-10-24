#pragma once

#include "types.h"
#include "accelerator/bvh.h"
#include "math/mat4.h"
#include "geometry/object.h"
#include "deformable/deformable.h"
#include "scene/context.h"

namespace aten
{
    template <typename OBJ>
    class instance : public transformable {
        friend class TransformableFactory;

    private:
        instance(OBJ* obj, const context& ctxt)
            : transformable(GeometryType::Instance)
        {
            m_obj = obj;
            setBoundingBox(m_obj->getBoundingbox());
        }

        instance(OBJ* obj, const context& ctxt, const mat4& mtxL2W)
            : instance(obj, ctxt)
        {
            m_mtxL2W = mtxL2W;

            m_mtxW2L = m_mtxL2W;
            m_mtxW2L.invert();

            setBoundingBox(getTransformedBoundingBox());
        }

        instance(
            OBJ* obj,
            const context& ctxt,
            const vec3& trans,
            const vec3& rot,
            const vec3& scale)
            : instance(obj, ctxt)
        {
            m_trans = trans;
            m_rot = rot;
            m_scale= scale;

            updateMatrix();

            m_mtxW2L = m_mtxL2W;
            m_mtxW2L.invert();

            setBoundingBox(getTransformedBoundingBox());
        }

        virtual ~instance() {}

    public:
        virtual bool hit(
            const context& ctxt,
            const ray& r,
            real t_min, real t_max,
            Intersection& isect) const override final
        {
            vec3 org = r.org;
            vec3 dir = r.dir;

            // Transform world to local.
            org = m_mtxW2L.apply(org);
            dir = m_mtxW2L.applyXYZ(dir);

            ray transformdRay(org, dir);

            // Hit test in local coordinate.
            auto isHit = m_obj->hit(ctxt, transformdRay, t_min, t_max, isect);

            if (isHit) {
                // returnTo this instance's id.
                isect.objid = id();
            }

            return isHit;
        }

        virtual void evalHitResult(
            const context& ctxt,
            const ray& r,
            hitrecord& rec,
            const Intersection& isect) const override final
        {
            m_obj->evalHitResult(ctxt, r, m_mtxL2W, rec, isect);

            // Transform local to world.
            rec.p = m_mtxL2W.apply(rec.p);
            rec.normal = normalize(m_mtxL2W.applyXYZ(rec.normal));

            rec.mtrlid = isect.mtrlid;
        }

        virtual void getSamplePosNormalArea(
            const context& ctxt,
            aten::hitable::SamplePosNormalPdfResult* result,
            sampler* sampler) const
        {
            return m_obj->getSamplePosNormalArea(ctxt, result, m_mtxL2W, sampler);
        }

        virtual const hitable* getHasObject() const override final
        {
            return m_obj;
        }

        OBJ* getHasObjectAsRealType()
        {
            return m_obj;
        }

        virtual const hitable* getHasSecondObject() const override final
        {
            return m_lod;
        }

        virtual void getMatrices(
            aten::mat4& mtxL2W,
            aten::mat4& mtxW2L) const override final
        {
            mtxL2W = m_mtxL2W;
            mtxW2L = m_mtxW2L;
        }

        virtual aabb getTransformedBoundingBox() const override
        {
            return std::move(aabb::transform(m_obj->getBoundingbox(), m_mtxL2W));
        }

        virtual void drawForGBuffer(
            aten::hitable::FuncPreDraw func,
            const context& ctxt,
            const aten::mat4& mtxL2W,
            const aten::mat4& mtxPrevL2W,
            int parentId,
            uint32_t triOffset) override final
        {
            m_obj->drawForGBuffer(func, ctxt, m_mtxL2W, m_mtxPrevL2W, id(), triOffset);
        }

        virtual void drawAABB(
            aten::hitable::FuncDrawAABB func,
            const aten::mat4& mtxL2W) override final
        {
            m_obj->drawAABB(func, m_mtxL2W);
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

        void setLod(OBJ* obj)
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
        virtual void getSamplePosNormalArea(
            const context& ctxt,
            aten::hitable::SamplePosNormalPdfResult* result,
            const mat4& mtxL2W,
            sampler* sampler) const override final
        {
            // Not used...
            AT_ASSERT(false);
        }

        virtual void evalHitResult(
            const context& ctxt,
            const ray& r,
            const mat4& mtxL2W,
            hitrecord& rec,
            const Intersection& isect) const override final
        {
            // Not used...
            AT_ASSERT(false);
        }

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
            m_mtxPrevL2W = m_mtxL2W;

            m_mtxL2W = mtxTrans * mtxRotX * mtxRotY * mtxRotZ * mtxScale;

            m_mtxW2L = m_mtxL2W;
            m_mtxW2L.invert();
        }

    private:
        OBJ* m_obj{ nullptr };
        OBJ* m_lod{ nullptr };

        mat4 m_mtxL2W;
        mat4 m_mtxW2L;    // inverted.
        mat4 m_mtxPrevL2W;

        vec3 m_trans;
        vec3 m_rot;
        vec3 m_scale{ aten::vec3(1, 1, 1) };

        bool m_isDirty{ false };
    };

    template<>
    inline instance<object>::instance(object* obj, const context& ctxt)
        : transformable(GeometryType::Instance)
    {
        m_obj = obj;
        m_obj->build(ctxt);
        setBoundingBox(m_obj->getBoundingbox());

        m_param.shapeid = ctxt.findTransformableIdxFromPointer(obj);
    }

    template<>
    inline instance<deformable>::instance(deformable* obj, const context& ctxt)
        : transformable(GeometryType::Instance)
    {
        m_obj = obj;
        m_obj->build();
        setBoundingBox(m_obj->getBoundingbox());

        m_param.shapeid = ctxt.findTransformableIdxFromPointer(obj);
    }
}
