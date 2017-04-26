#pragma once

#include <vector>
#include "scene/accel.h"
#include "scene/bvh.h"
#include "light/light.h"
#include "light/ibl.h"

namespace AT_NAME {
	class LinearList : public aten::accel {
	public:
		LinearList() {}
		~LinearList() {}

		virtual void build(
			aten::bvhnode** list,
			uint32_t num) override final
		{
			for (uint32_t i = 0; i < num; i++) {
				m_objs.push_back(list[i]);
			}
		}

		virtual aten::aabb getBoundingbox() const override final
		{
			// TODO
			AT_ASSERT(false);
			return std::move(aten::aabb());
		}

		virtual bool hit(
			const aten::ray& r,
			real t_min, real t_max,
			aten::hitrecord& rec) const override final
		{
			bool isHit = false;

			aten::hitrecord tmp;

			for (size_t i = 0; i < m_objs.size(); i++) {
				auto o = m_objs[i];
				if (o->hit(r, t_min, t_max, tmp)) {
					if (tmp.t < rec.t) {
						rec = tmp;
						rec.obj = o;

						t_max = tmp.t;

						isHit = true;
					}
				}
			}

			return isHit;
		}

	private:
		std::vector<aten::bvhnode*> m_objs;
	};

	class scene {
	public:
		scene() {}
		virtual ~scene() {}

	public:
		virtual void build()
		{}

		void add(aten::bvhnode* s)
		{
			m_tmp.push_back(s);
		}

		virtual bool hit(
			const aten::ray& r,
			real t_min, real t_max,
			aten::hitrecord& rec) const = 0;

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

		template <typename Func>
		static AT_DEVICE_API bool hitLight(
			Func funcHitTest,
			const aten::LightParameter& light,
			const aten::vec3& lightPos,
			const aten::ray& r,
			real t_min, real t_max,
			aten::hitrecord& rec)
		{
			bool isHit = funcHitTest(r, t_min, t_max, rec);

			auto lightobj = light.object.ptr;

			if (lightobj) {
				// Area Light.
				if (isHit) {
#if 0
					hitrecord tmpRec;
					if (lightobj->hit(r, t_min, t_max, tmpRec)) {
						auto dist2 = (tmpRec.p - r.org).squared_length();

						if (rec.obj == tmpRec.obj
							&& aten::abs(dist2 - rec.t * rec.t) < AT_MATH_EPSILON)
						{
							return true;
						}
					}
#else
					auto dist = (rec.p - r.org).length();

					if (rec.obj == lightobj
						&& aten::abs(dist - rec.t) <= AT_MATH_EPSILON)
					{
						return true;
					}
#endif
				}
			}

			if (light.attrib.isInfinite) {
				if (isHit) {
					// Hit something.
					return false;
				}
				else {
					// Hit nothing.
					return true;
				}
			}
			else if (light.attrib.isSingular) {
				auto distToLight = (lightPos - r.org).length();

				if (isHit && rec.t < distToLight) {
					// Ray hits something, and the distance to the object is near than the distance to the light.
					return false;
				}
				else {
					// Ray don't hit anything, or the distance to the object is far than the distance to the light.
					return true;
				}
			}

			return false;
		}

		Light* sampleLight(
			const aten::vec3& org,
			const aten::vec3& nml,
			aten::sampler* sampler,
			real& selectPdf,
			aten::LightSampleResult& sampleRes);

	protected:
		std::vector<aten::bvhnode*> m_tmp;

		std::vector<Light*> m_lights;
		ImageBasedLight* m_ibl{ nullptr };
	};

	template <typename ACCEL>
	class AcceledScene : public scene {
	public:
		AcceledScene() {}
		virtual ~AcceledScene() {}

	public:
		virtual void build() override final
		{
			if (!m_tmp.empty()) {
				m_accel.build(&m_tmp[0], (uint32_t)m_tmp.size());
			}
		}

		virtual bool hit(
			const aten::ray& r,
			real t_min, real t_max,
			aten::hitrecord& rec) const override final
		{
			auto isHit = m_accel.hit(r, t_min, t_max, rec);
			return isHit;
		}

		const accel* getAccel()
		{
			return &m_accel;
		}

	private:
		ACCEL m_accel;
	};
}
