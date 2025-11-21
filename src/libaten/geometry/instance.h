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
    template <class OBJ>
    class instance : public transformable {
        friend class TransformableFactory;

    public:
        instance(const std::shared_ptr<OBJ>& obj, context& ctxt)
            : transformable(ObjectType::Instance), m_obj(obj)
        {
            init_matrices(ctxt);

            m_param.object_id = m_obj->id();
            m_obj->build(ctxt, std::nullopt);
            setBoundingBox(m_obj->getBoundingbox());

            setBoundingBox(getTransformedBoundingBox());
        }

        instance(const std::shared_ptr<OBJ>& obj, context& ctxt, const mat4& mtx_L2W)
            : transformable(ObjectType::Instance), m_obj(obj)
        {
            init_matrices(ctxt);

            aten::vec3 scale;
            aten::mat4 mtx_rot;
            aten::vec3 trans;
            aten::tie(scale, mtx_rot, trans) = mtx_L2W.Decompose();

            aten::mat4 mtx_trans;
            mtx_trans.asTrans(trans);

            *m_mtx_L2W = mtx_trans * mtx_rot;

            *m_mtx_W2L = *m_mtx_L2W;
            m_mtx_W2L->invert();

            const bool no_scale = CheckNoScale(scale);

            m_param.object_id = m_obj->id();
            m_obj->build(
                ctxt,
                no_scale ? std::nullopt : std::make_optional(scale));
            setBoundingBox(m_obj->getBoundingbox());

            setBoundingBox(getTransformedBoundingBox());
        }

        instance(
            const std::shared_ptr<OBJ>& obj,
            context& ctxt,
            const vec3& trans,
            const vec3& rot,
            const vec3& scale)
            : transformable(ObjectType::Instance), m_obj(obj)
        {
            init_matrices(ctxt);

            aten::mat4 mtx_trans;
            mtx_trans.asTrans(trans);

            aten::mat4 mtx_rot_x;
            mtx_rot_x.asRotateByX(rot.x);

            aten::mat4 mtx_rot_y;
            mtx_rot_y.asRotateByX(rot.y);

            aten::mat4 mtx_rot_z;
            mtx_rot_z.asRotateByX(rot.z);

            // TODO:
            // We still doesn't support scaling in bvh traverse.
            // So, scale is ignored to construct local to world matrix for bvh traverse.
            *m_mtx_L2W = mtx_trans * mtx_rot_x * mtx_rot_y * mtx_rot_z;

            *m_mtx_W2L = *m_mtx_L2W;
            m_mtx_W2L->invert();

            const bool no_scale = CheckNoScale(scale);

            m_param.object_id = m_obj->id();
                m_obj->build(
                    ctxt,
                    no_scale ? std::nullopt : std::make_optional(scale));
            setBoundingBox(m_obj->getBoundingbox());

            setBoundingBox(getTransformedBoundingBox());
        }

        virtual ~instance() = default;

        virtual bool hit(
            const context& ctxt,
            const ray& r,
            float t_min, float t_max,
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

        std::shared_ptr<OBJ> GetHasObjectAsRealType()
        {
            return m_obj;
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

        virtual void update(aten::context& ctxt, bool isForcibly = false)
        {
            if (m_isDirty || isForcibly) {
                // TODO:
                // Not implement ApplyScaleToObj in PolygonObject and deformable classes yet...
                ApplyScaleToObj(ctxt);

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
            // NOTE:
            // L2W matrix and W2L matrix are treated as pair.
            // And, the matrices are listed as pair in the list.
            //
            // e.g.
            // idx:   0      1      2      3      4      5    .....
            // mtx: L2W_0, W2L_0, L2W_1, W2L_1, L2W_2, W2L_2, .....
            //
            // It means, if the index to one of them is identified, the index to another is just +1.
            // e.g. If the index to mtx_L2W is 2. the index to mtx_W2L is 3.
            // So, just keeping the index to one variable mtx_id is enough. No need to keep the two indices.

            aten::tie(m_param.mtx_id, m_mtx_L2W) = ctxt.CreateMatrix();
            aten::tie(std::ignore, m_mtx_W2L) = ctxt.CreateMatrix();
        }

        static bool CheckNoScale(const aten::vec3& scale)
        {
            return aten::isClose(scale.x, 1.0F)
                && aten::isClose(scale.y, 1.0F)
                && aten::isClose(scale.z, 1.0F);
        }

        void ApplyScaleToObj(aten::context& ctxt)
        {
            // TODO:
            // Not sure how we should do for LOD yet....
            // So, if lod is available, raise the assertion.
            AT_ASSERT(!m_lod);

            if (m_obj && !m_lod) {
                const auto no_scale = CheckNoScale(m_scale);
                if (!no_scale) {
                    // TODO:
                    // Not implement ApplyScaleToObj in PolygonObject and deformable classes yet...
                    //m_obj->ApplyScaleToObj(ctxt, m_scale);
                }
            }
        }

    private:
        std::shared_ptr<OBJ> m_obj;
        std::shared_ptr<OBJ> m_lod;

        std::shared_ptr<mat4> m_mtx_L2W;
        std::shared_ptr<mat4> m_mtx_W2L;    // inverted.
        mat4 m_mtx_prev_L2W;

        vec3 m_trans{ 0.0F };
        vec3 m_rot{ 0.0F };
        vec3 m_scale{ 1.0F };

        bool m_isDirty{ false };
    };
}
