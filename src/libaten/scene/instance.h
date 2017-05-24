#pragma once

#include "types.h"
#include "accelerator/bvh.h"
#include "math/mat4.h"
#include "object/object.h"

namespace aten
{
	template <typename OBJ>
	class instance : public transformable {
	public:
		instance()
			: m_param(ShapeType::Instance)
		{}

		instance(OBJ* obj)
			: m_param(ShapeType::Instance)
		{
			m_obj = obj;
			m_aabb = m_obj->getBoundingbox();
		}

		instance(OBJ* obj, const mat4& mtxL2W)
			: instance(obj)
		{
			m_mtxL2W = mtxL2W;

			m_mtxW2L = m_mtxL2W;
			m_mtxW2L.invert();

			m_aabb = transformBoundingBox();

			m_param.mtxL2W = m_mtxL2W;
			m_param.mtxW2L = m_mtxW2L;
		}

		virtual ~instance() {}

	public:
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override final
		{
			vec3 org = r.org;
			vec3 dir = r.dir;

			// Transform world to local.
			org = m_mtxW2L.apply(org);
			dir = m_mtxW2L.applyXYZ(dir);

			ray transformdRay(org, dir);

			// Hit test in local coordinate.
			auto isHit = m_obj->hit(transformdRay, t_min, t_max, rec);

			if (isHit) {
				rec.obj = (hitable*)this;
			}

			return isHit;
		}

		virtual void evalHitResult(
			const ray& r,
			hitrecord& rec) const override final
		{
			m_obj->evalHitResult(r, m_mtxL2W, rec);

			// Transform local to world.
			rec.p = m_mtxL2W.apply(rec.p);
			rec.normal = normalize(m_mtxL2W.applyXYZ(rec.normal));

			// tangent coordinate.
			rec.du = normalize(getOrthoVector(rec.normal));
			rec.dv = normalize(cross(rec.normal, rec.du));
		}

		virtual void getSamplePosNormalArea(
			aten::hitable::SamplePosNormalPdfResult* result,
			sampler* sampler) const
		{
			return m_obj->getSamplePosNormalArea(result, m_mtxL2W, sampler);
		}

		virtual const hitable* getHasObject() const override final
		{
			return m_obj;
		}

	private:
		aabb transformBoundingBox()
		{
			return std::move(aabb::transform(m_aabb, m_mtxL2W));
		}

		virtual void getSamplePosNormalArea(
			aten::hitable::SamplePosNormalPdfResult* result,
			const mat4& mtxL2W,
			sampler* sampler) const override final
		{
			// Not used...
			AT_ASSERT(false);
		}

		virtual void evalHitResult(
			const ray& r,
			const mat4& mtxL2W,
			hitrecord& rec) const override final
		{
			// Not used...
			AT_ASSERT(false);
		}

		virtual const ShapeParameter& getParam() const override final
		{
			return m_param;
		}

		virtual int collectInternalNodes(
			std::vector<std::vector<BVHNode>>& nodes, 
			int order, 
			bvhnode* parent,
			const aten::mat4& mtxL2W) override final
		{
			return m_obj->collectInternalNodes(nodes, order, this, m_mtxL2W);
		}

	private:
		OBJ* m_obj{ nullptr };
		mat4 m_mtxL2W;
		mat4 m_mtxW2L;	// inverted.

		ShapeParameter m_param;
	};

	template<>
	instance<object>::instance(object* obj)
		: m_param(ShapeType::Instance)
	{
		m_obj = obj;
		m_obj->build();
		m_aabb = m_obj->bbox;

		m_param.shapeid = transformable::findShapeIdx(obj);
	}
}
