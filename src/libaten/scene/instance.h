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
		}

		virtual ~instance() {}

	public:
		virtual bool hit(
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
			auto isHit = m_obj->hit(transformdRay, t_min, t_max, isect);

			if (isHit) {
				// Return this instance's id.
				isect.objid = id();
			}

			return isHit;
		}

		virtual void evalHitResult(
			const ray& r,
			hitrecord& rec,
			const Intersection& isect) const override final
		{
			m_obj->evalHitResult(r, m_mtxL2W, rec, isect);

			// Transform local to world.
			rec.p = m_mtxL2W.apply(rec.p);
			rec.normal = normalize(m_mtxL2W.applyXYZ(rec.normal));

			rec.objid = isect.objid;
			rec.mtrlid = isect.mtrlid;
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

		virtual void getMatrices(
			aten::mat4& mtxL2W,
			aten::mat4& mtxW2L) const override final
		{
			mtxL2W = m_mtxL2W;
			mtxW2L = m_mtxW2L;
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
			hitrecord& rec,
			const Intersection& isect) const override final
		{
			// Not used...
			AT_ASSERT(false);
		}

		virtual const ShapeParameter& getParam() const override final
		{
			return m_param;
		}

		virtual bool setBVHNodeParam(
			BVHNode& param,
			const bvhnode* parent,
			const int idx,
			std::vector<std::vector<BVHNode>>& nodes,
			const transformable* instanceParent,
			const aten::mat4& mtxL2W) override final
		{
			m_obj->setBVHNodeParam(param, parent, idx, nodes, this, m_mtxL2W);
			return false;
		}

		virtual void registerToList(
			const int idx,
			std::vector<std::vector<bvhnode*>>& nodeList) override final
		{
			m_obj->registerToList(idx, nodeList);
		}

		virtual bvhnode* getNode() override final
		{
			return m_obj->getNode();
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
