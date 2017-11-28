#pragma once

#include <vector>
#include "accelerator/accelerator.h"
#include "accelerator/bvh.h"
#include "light/light.h"
#include "light/ibl.h"

namespace AT_NAME {
	class scene {
	public:
		scene() {}
		virtual ~scene() {}

	public:
		virtual void build()
		{}

		void add(aten::hitable* s)
		{
			m_tmp.push_back(s);
		}

		virtual bool hit(
			const aten::ray& r,
			real t_min, real t_max,
			aten::hitrecord& rec,
			aten::Intersection& isect) const = 0;

		virtual bool hit(
			const aten::accelerator::ResultIntersectTestByFrustum& resF,
			const aten::ray& r,
			real t_min, real t_max,
			aten::hitrecord& rec,
			aten::Intersection& isect) const = 0;

		void addLight(Light* l)
		{
			m_lights.push_back(l);
		}

		void addImageBasedLight(ImageBasedLight* l)
		{
			if (m_ibl != l) {
				m_ibl = l;

				// TODO
				// Remove light, before adding.
				addLight(l);
			}
		}

		uint32_t lightNum() const
		{
			return (uint32_t)m_lights.size();
		}

		// TODO
		Light* getLight(uint32_t i)
		{
			if (m_lights.empty()) {
				return nullptr;
			}

			if (i >= lightNum()) {
				AT_ASSERT(false);
				return nullptr;
			}
			return m_lights[i];
		}

		ImageBasedLight* getIBL()
		{
			return m_ibl;
		}

		bool hitLight(
			const Light* light,
			const aten::vec3& lightPos,
			const aten::ray& r,
			real t_min, real t_max,
			aten::hitrecord& rec);

		static inline AT_DEVICE_API bool hitLight(
			bool isHit,
			aten::LightAttribute attrib,
			const void* lightobj,
			real distToLight,
			real distHitObjToRayOrg,
			const real hitt,
			const void* hitobj)
		{
#if 0
			//auto lightobj = light->object.ptr;

			if (lightobj) {
				// Area Light.
				if (isHit) {
#if 0
					hitrecord tmpRec;
					if (lightobj->hit(r, t_min, t_max, tmpRec)) {
						auto dist2 = squared_length(tmpRec.p - r.org);

						if (rec->obj == tmpRec.obj
							&& aten::abs(dist2 - rec->t * rec->t) < AT_MATH_EPSILON)
						{
							return true;
						}
					}
#else
					//auto distHitObjToRayOrg = (hitp - r.org).length();

					if (hitobj == lightobj
						&& aten::abs(distHitObjToRayOrg - hitt) <= AT_MATH_EPSILON)
					{
						return true;
					}
#endif
				}
			}

			if (attrib.isInfinite) {
				return !isHit;
			}
			else if (attrib.isSingular) {
				//auto distToLight = (lightPos - r.org).length();

				if (isHit && hitt < distToLight) {
					// Ray hits something, and the distance to the object is near than the distance to the light.
					return false;
				}
				else {
					// Ray don't hit anything, or the distance to the object is far than the distance to the light.
					return true;
				}
			}

			return false;
#else
			//if (isHit && hitobj == lightobj) {
			if (hitobj == lightobj) {
				return true;
			}

			if (attrib.isInfinite) {
				return !isHit;
			}
			else if (attrib.isSingular) {
				return hitt > distToLight;
			}

			return false;
#endif
		}

		Light* sampleLight(
			const aten::vec3& org,
			const aten::vec3& nml,
			aten::sampler* sampler,
			real& selectPdf,
			aten::LightSampleResult& sampleRes);

		void draw(aten::hitable::FuncPreDraw func);

	protected:
		std::vector<aten::hitable*> m_tmp;

		std::vector<Light*> m_lights;
		ImageBasedLight* m_ibl{ nullptr };
	};

	template <typename ACCEL>
	class AcceleratedScene : public scene {
	public:
		AcceleratedScene()
		{
			accelerator::setInternalAccelType(m_accel.getAccelType());
		}
		virtual ~AcceleratedScene() {}

	public:
		virtual void build() override final
		{
			aten::aabb bbox;

			for (const auto& t : m_tmp) {
				bbox = aten::aabb::merge(bbox, t->getBoundingbox());
			}

			if (!m_tmp.empty()) {
				m_accel.build(&m_tmp[0], (uint32_t)m_tmp.size(), &bbox);
			}
		}

		virtual bool hit(
			const aten::ray& r,
			real t_min, real t_max,
			aten::hitrecord& rec,
			aten::Intersection& isect) const override final
		{
			auto isHit = m_accel.hit(r, t_min, t_max, isect);

			// TODO
#ifndef __AT_CUDA__
			if (isHit) {
				auto obj = transformable::getShape(isect.objid);
				aten::hitable::evalHitResult(obj, r, rec, isect);
			}
#endif

			return isHit;
		}

		virtual bool hit(
			const aten::accelerator::ResultIntersectTestByFrustum& resF,
			const aten::ray& r,
			real t_min, real t_max,
			aten::hitrecord& rec,
			aten::Intersection& isect) const override final
		{
			auto isHit = m_accel.hitMultiLevel(resF, r, t_min, t_max, isect);

			// TODO
#ifndef __AT_CUDA__
			if (isHit) {
				auto obj = transformable::getShape(isect.objid);
				aten::hitable::evalHitResult(obj, r, rec, isect);
			}
#endif

			return isHit;

		}

		ACCEL* getAccel()
		{
			return &m_accel;
		}

	private:
		ACCEL m_accel;
	};
}
