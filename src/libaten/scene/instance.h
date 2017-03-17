#pragma once

#include "types.h"
#include "scene/bvh.h"
#include "math/mat4.h"
#include "object/object.h"

namespace aten
{
	template <typename OBJ>
	class instance : public bvhnode {
	public:
		instance() {}

		instance(OBJ* obj)
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
			hitrecord& rec) const override final
		{
			vec3 org = r.org;
			vec3 dir = r.dir;

			org = m_mtxW2L.apply(org);
			dir = m_mtxW2L.applyXYZ(dir);

			ray transformdRay(org, dir);

			auto isHit = m_obj->hit(transformdRay, t_min, t_max, rec);

			if (isHit) {
				// Transform local to world.
				rec.p = m_mtxL2W.apply(rec.p);
				rec.normal = normalize(m_mtxL2W.applyXYZ(rec.normal));
			}

			return isHit;
		}

		virtual aabb getBoundingbox() const override final
		{
			return std::move(m_aabb);
		}

	private:
		aabb transformBoundingBox()
		{
			const auto& box = m_aabb;

			vec3 center = box.getCenter();

			vec3 vMin = box.minPos() - center;
			vec3 vMax = box.maxPos() - center;

			vec3 pts[8] = {
				vec3(vMin.x, vMin.y, vMin.z),
				vec3(vMax.x, vMin.y, vMin.z),
				vec3(vMin.x, vMax.y, vMin.z),
				vec3(vMax.x, vMax.y, vMin.z),
				vec3(vMin.x, vMin.y, vMax.z),
				vec3(vMax.x, vMin.y, vMax.z),
				vec3(vMin.x, vMax.y, vMax.z),
				vec3(vMax.x, vMax.y, vMax.z),
			};

			vec3 newMin(AT_MATH_INF);
			vec3 newMax(-AT_MATH_INF);

			for (int i = 0; i < 8; i++) {
				vec3 v = m_mtxL2W.apply(pts[i]);

				newMin.set(
					std::min(newMin.x, v.x),
					std::min(newMin.y, v.y),
					std::min(newMin.z, v.z));
				newMax.set(
					std::max(newMax.x, v.x),
					std::max(newMax.y, v.y),
					std::max(newMax.z, v.z));
			}

			// Add only translate factor.
			vec3 newCenter = center + m_mtxL2W.apply(vec3());

			newMin += newCenter;
			newMax += newCenter;

			aabb ret(newMin, newMax);

			return std::move(ret);
		}

	private:
		OBJ* m_obj{ nullptr };
		mat4 m_mtxL2W;
		mat4 m_mtxW2L;	// inverted.
	};

	template<>
	instance<object>::instance(object* obj)
	{
		m_obj = obj;
		m_obj->build();
		m_aabb = m_obj->bbox;
	}
}
