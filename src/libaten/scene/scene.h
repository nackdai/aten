#pragma once

#include <vector>
#include "scene/accel.h"
#include "scene/bvh.h"
#include "light/light.h"
#include "light/ibl.h"

namespace aten {
	class LinearList : public accel {
	public:
		LinearList() {}
		~LinearList() {}

		virtual void build(
			bvhnode** list,
			uint32_t num) override final
		{
			for (uint32_t i = 0; i < num; i++) {
				m_objs.push_back(list[i]);
			}
		}

		virtual aabb getBoundingbox() const override final
		{
			// TODO
			AT_ASSERT(false);
			return std::move(aabb());
		}

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override final
		{
			bool isHit = false;

			hitrecord tmp;

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
		std::vector<bvhnode*> m_objs;
	};

	class scene {
	public:
		scene() {}
		virtual ~scene() {}

		void add(bvhnode* s)
		{
			m_tmp.push_back(s);
		}

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const = 0;

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
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec);

		Light* sampleLight(
			const vec3& org,
			const vec3& nml,
			sampler* sampler,
			real& selectPdf,
			LightSampleResult& sampleRes);

	protected:
		std::vector<bvhnode*> m_tmp;

		std::vector<Light*> m_lights;
		ImageBasedLight* m_ibl{ nullptr };
	};

	template <typename ACCEL>
	class AcceledScene : public scene {
	public:
		AcceledScene() {}
		virtual ~AcceledScene() {}

	public:
		void build()
		{
			if (!m_tmp.empty()) {
				m_accel.build(&m_tmp[0], (uint32_t)m_tmp.size());
			}
		}

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override final
		{
			auto isHit = m_accel.hit(r, t_min, t_max, rec);
			return isHit;
		}

	private:
		ACCEL m_accel;
	};
}
